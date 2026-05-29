#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Launch examples:

- single available GPU (default)
python -m scripts.chat_web

- 4 GPUs
python -m scripts.chat_web --num-gpus 4

To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)

Endpoints:
  GET  /           - Chat UI
  POST /chat/completions - Chat API (streaming only)
  GET  /health     - Health check with worker pool status
  GET  /stats      - Worker pool statistics and GPU utilization

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 0-200 (0 disables top-k filtering, using full vocabulary)
  - Max tokens clamped to 1-4096
"""

import argparse
import base64
import io
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from nanochat.common import COMPUTE_DTYPE, compute_init, autodetect_device_type
from nanochat.checkpoint_manager import _patch_missing_config_keys, _patch_missing_keys, load_model
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.vision import (
    IMAGE_MARKER,
    SIGLIP_MODEL_ID,
    SigLIPPooledFeatureExtractor,
    VisionProjector,
    build_multimodal_batch,
    encode_with_image_markers,
    format_image_markers,
)

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0 # 0 disables top-k filtering, using full vocabulary
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096
MAX_IMAGES_PER_REQUEST = 8
MAX_IMAGE_BYTES = 8 * 1024 * 1024

parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
parser.add_argument('--vlm-checkpoint-dir', default=None, help='Optional VLM checkpoint directory; enables image chat')
parser.add_argument('--vlm-checkpoint-step', type=int, default=None, help='VLM checkpoint step')
parser.add_argument('--llm-checkpoint-dir', default=None, help='Optional language-model checkpoint directory to combine with a VLM projector')
parser.add_argument('--llm-checkpoint-step', type=int, default=None, help='Language-model checkpoint step when --llm-checkpoint-dir is set')
parser.add_argument('--siglip-model-id', default=SIGLIP_MODEL_ID, help='SigLIP model used by the VLM checkpoint')
parser.add_argument('--siglip-cache-dir', default=None, help='Optional SigLIP cache directory')
parser.add_argument('--eval-dtype', default='', choices=['', 'float32', 'float16', 'bfloat16'], help='Optional dtype for loading eval weights. Empty follows NANOCHAT_DTYPE/auto detection.')
args = parser.parse_args()

# Configure logging for conversation traffic
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_eval_dtype():
    return DTYPE_MAP[args.eval_dtype] if args.eval_dtype else COMPUTE_DTYPE


def cast_floating_tensors_in_place(state_dict, dtype):
    for key, value in list(state_dict.items()):
        if torch.is_tensor(value) and value.is_floating_point() and value.dtype != dtype:
            state_dict[key] = value.to(dtype=dtype)
    return state_dict


def build_empty_gpt(model_config, device, dtype):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = GPT(model_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.to_empty(device=device)
    head_dim = model_config.n_embd // model_config.n_head
    model.cos, model.sin = model._precompute_rotary_embeddings(model.rotary_seq_len, head_dim, device=device)
    return model


def load_gpt_model_direct(checkpoint_dir, step, device, target_dtype):
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    model_state = torch.load(model_path, map_location="cpu", mmap=True)
    model_state = {k.removeprefix("_orig_mod."): v for k, v in model_state.items()}

    model_config_kwargs = dict(meta_data["model_config"])
    _patch_missing_config_keys(model_config_kwargs)
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_state, model_config)
    cast_floating_tensors_in_place(model_state, target_dtype)

    print(f"Building GPT model directly from {checkpoint_dir}@{step} with dtype={target_dtype}", flush=True)
    model = build_empty_gpt(model_config, device, target_dtype)
    model.load_state_dict(model_state, strict=True)
    del model_state

    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size "
        f"{model_config_kwargs['vocab_size']}"
    )
    return model, tokenizer, meta_data


def load_vlm_model_direct(checkpoint_dir, step, device, llm_checkpoint_dir=None, llm_checkpoint_step=None):
    """Load a VLM checkpoint without first materializing the base SFT model."""
    target_dtype = resolve_eval_dtype()
    if args.eval_dtype and target_dtype != COMPUTE_DTYPE:
        print(
            f"Warning: --eval-dtype={args.eval_dtype} only controls stored weights; "
            f"activations still use NANOCHAT_DTYPE/auto={COMPUTE_DTYPE}.",
            flush=True,
        )

    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    model_data = torch.load(model_path, map_location="cpu", mmap=True)
    projector_state = model_data["projector"]
    projector_config = model_data["projector_config"]

    if llm_checkpoint_dir is None:
        model_state = {k.removeprefix("_orig_mod."): v for k, v in model_data["model"].items()}
        model_config_kwargs = dict(meta_data["model_config"])
        _patch_missing_config_keys(model_config_kwargs)
        model_config = GPTConfig(**model_config_kwargs)
        _patch_missing_keys(model_state, model_config)
        cast_floating_tensors_in_place(model_state, target_dtype)

        print(f"Building VLM model directly from {checkpoint_dir}@{step} with dtype={target_dtype}", flush=True)
        model = build_empty_gpt(model_config, device, target_dtype)
        model.load_state_dict(model_state, strict=True)
        del model_state

        tokenizer = get_tokenizer()
        assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], (
            f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size "
            f"{model_config_kwargs['vocab_size']}"
        )
    else:
        assert llm_checkpoint_step is not None, "--llm-checkpoint-step is required with --llm-checkpoint-dir"
        model, tokenizer, llm_meta = load_gpt_model_direct(llm_checkpoint_dir, llm_checkpoint_step, device, target_dtype)
        meta_data = {
            "vlm_projector_meta": meta_data,
            "llm_meta": llm_meta,
            "hybrid": True,
        }

    cast_floating_tensors_in_place(projector_state, target_dtype)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        projector = VisionProjector(**projector_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    projector.load_state_dict(projector_state, strict=True)
    projector.to(device=device, dtype=target_dtype)
    del projector_state, model_data
    return model, tokenizer, projector, meta_data

@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""
    gpu_id: int
    device: torch.device
    model: object
    engine: Engine
    tokenizer: object
    projector: object = None
    extractor: object = None

class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."
        if args.vlm_checkpoint_dir is not None:
            assert args.vlm_checkpoint_step is not None, "--vlm-checkpoint-step is required with --vlm-checkpoint-dir"
        if args.llm_checkpoint_dir is not None:
            assert args.vlm_checkpoint_dir is not None, "--llm-checkpoint-dir is only valid with --vlm-checkpoint-dir"
            assert args.llm_checkpoint_step is not None, "--llm-checkpoint-step is required with --llm-checkpoint-dir"

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                print(f"Loading model on {device_type}...")

            projector = None
            extractor = None
            if args.vlm_checkpoint_dir is not None:
                print(f"Loading VLM checkpoint on {device}: {args.vlm_checkpoint_dir}@{args.vlm_checkpoint_step}")
                if args.llm_checkpoint_dir is not None:
                    print(f"Using language model from {args.llm_checkpoint_dir}@{args.llm_checkpoint_step}")
                model, tokenizer, projector, _ = load_vlm_model_direct(
                    args.vlm_checkpoint_dir,
                    args.vlm_checkpoint_step,
                    device,
                    llm_checkpoint_dir=args.llm_checkpoint_dir,
                    llm_checkpoint_step=args.llm_checkpoint_step,
                )
                projector.eval()
                siglip_cache_dir = args.siglip_cache_dir or os.environ.get("NANOCHAT_SIGLIP_CACHE_DIR")
                extractor = SigLIPPooledFeatureExtractor(
                    args.siglip_model_id,
                    device=device,
                    dtype=resolve_eval_dtype() if device.type in {"cuda", "mps"} else None,
                    cache_dir=siglip_cache_dir,
                    verbose=gpu_id == 0,
                )
            else:
                model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            model.eval()
            engine = Engine(model, tokenizer)
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                model=model,
                engine=engine,
                tokenizer=tokenizer,
                projector=projector,
                extractor=extractor,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str
    image: Optional[str] = None
    images: Optional[List[str]] = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    # Check number of messages
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    # Check individual message lengths and total conversation length
    total_length = 0
    image_count = 0
    for i, message in enumerate(request.messages):
        message_images = []
        if message.image:
            message_images.append(message.image)
        if message.images:
            message_images.extend(message.images)
        if not message.content and not message_images:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        total_length += msg_length
        if message_images:
            if message.role != "user":
                raise HTTPException(status_code=400, detail=f"Message {i} has images but is not a user message")
            image_count += len(message_images)
            for image_idx, image_data in enumerate(message_images):
                if len(image_data) > MAX_IMAGE_BYTES * 2:
                    raise HTTPException(status_code=400, detail=f"Message {i} image {image_idx} is too large")

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
        )
    if image_count > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many images. Maximum {MAX_IMAGES_PER_REQUEST} images allowed")

    # Validate role values
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'"
            )

    # Validate temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            )

    # Validate top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )

    # Validate max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
            )

def decode_image_data(image_data: str):
    """Decode a browser data URL into a RGB PIL image."""
    if image_data.startswith("data:"):
        try:
            _, image_data = image_data.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid image encoding")
    try:
        raw = base64.b64decode(image_data, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image encoding")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image is too large")
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on all GPUs on startup."""
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"Server ready at http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the chat UI."""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Replace the API_URL to use the same origin
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """Serve the NanoChat logo for favicon and header."""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # Accumulate tokens to properly handle multi-byte UTF-8 characters (like emojis)
    accumulated_tokens = []
    # Track the last complete UTF-8 string (without replacement characters)
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1)
    ):
        token = token_column[0]

        # Stopping criteria
        if token == assistant_end or token == bos:
            break

        # Append the token to sequence
        accumulated_tokens.append(token)
        # Decode all accumulated tokens to get proper UTF-8 handling
        # Note that decode is a quite efficient operation, basically table lookup and string concat
        current_text = worker.tokenizer.decode(accumulated_tokens)
        # Only emit text if it doesn't end with a replacement character
        # This ensures we don't emit incomplete UTF-8 sequences
        if not current_text.endswith('�'):
            # Extract only the new text since last clean decode
            new_text = current_text[len(last_clean_text):]
            if new_text:  # Only yield if there's new content
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

async def generate_vision_stream(
    worker: Worker,
    tokens,
    images,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """Generate assistant response from a multimodal prompt with KV-cache decode."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    image_features = worker.extractor(images)
    row = list(tokens) + [tokens[-1]]
    mask = [0] * len(row)
    batch = build_multimodal_batch(
        worker.model,
        worker.projector,
        [row],
        image_features,
        loss_mask_rows=[mask],
        image_counts_per_row=[len(images)],
        value_fallback_token_id=bos,
    )
    accumulated_tokens = []
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
        prefill_input_embeds=batch.input_embeds,
        prefill_value_token_ids=batch.value_token_ids,
        prefill_length=int(batch.lengths[0].item()),
    ):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break

        accumulated_tokens.append(token)
        current_text = worker.tokenizer.decode(accumulated_tokens)
        if not current_text.endswith('�'):
            new_text = current_text[len(last_clean_text):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text
        await asyncio.sleep(0)

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) - uses worker pool for multi-GPU."""

    # Basic validation to prevent abuse
    validate_chat_request(request)

    # Log incoming conversation to console
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        message_image_count = int(message.image is not None) + (len(message.images) if message.images else 0)
        image_suffix = f" [{message_image_count} image{'s' if message_image_count != 1 else ''}]" if message_image_count else ""
        logger.info(f"[{message.role.upper()}]{image_suffix}: {message.content}")
    logger.info("-"*20)

    # Acquire a worker from the pool (will wait if all are busy)
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # Build conversation tokens
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        images = []
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                message_images = []
                if message.image:
                    message_images.append(message.image)
                if message.images:
                    message_images.extend(message.images)
                if message_images:
                    if worker.projector is None or worker.extractor is None:
                        raise HTTPException(status_code=400, detail="Server was not started with --vlm-checkpoint-dir")
                    images.extend(decode_image_data(image_data) for image_data in message_images)
                    content = message.content.replace(IMAGE_MARKER, "").strip()
                    content = content or ("Describe the images." if len(message_images) > 1 else "Describe the image.")
                    image_markers = format_image_markers(len(message_images))
                    conversation_tokens.extend(encode_with_image_markers(worker.tokenizer, f"{image_markers}\n{content}"))
                else:
                    conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # Streaming response with worker release after completion
        response_tokens = []
        async def stream_and_release():
            try:
                generator = generate_vision_stream if images else generate_stream
                kwargs = {"images": images} if images else {}
                async for chunk in generator(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                    **kwargs,
                ):
                    # Accumulate response for logging
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # Log the assistant response to console
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # Release worker back to pool after streaming is done
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Make sure to release worker even on error
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """Health check endpoint."""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "vlm": args.vlm_checkpoint_dir is not None,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """Get worker pool statistics."""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device),
                "vlm": w.projector is not None,
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Web Server")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    if args.vlm_checkpoint_dir is not None:
        print(f"VLM image chat: {args.vlm_checkpoint_dir}@{args.vlm_checkpoint_step}")
        if args.llm_checkpoint_dir is not None:
            print(f"Hybrid LLM checkpoint: {args.llm_checkpoint_dir}@{args.llm_checkpoint_step}")
    uvicorn.run(app, host=args.host, port=args.port)

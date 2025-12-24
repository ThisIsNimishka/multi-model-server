"""
Multi-Model AI Server v3 - With Flash Attention 2 Support
=========================================================
Supports: Mistral-7B, Qwen2.5-VL-7B with optimized attention
Usage:
    python server_v3.py --model mistral --port 8001
    python server_v3.py --model qwen-vl --port 8002
"""

import argparse
import base64
import io
import logging
import os
import sys
import time
from typing import List, Optional, Any, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Check Flash Attention Availability
# ============================================
def check_flash_attention():
    """Check if Flash Attention 2 is available"""
    flash_attn_available = False
    sdpa_available = False
    
    # Check flash-attn package
    try:
        import flash_attn
        flash_attn_available = True
        logger.info(f"✓ Flash Attention 2 available (flash-attn {flash_attn.__version__})")
    except ImportError:
        logger.info("✗ flash-attn not installed")
    
    # Check PyTorch SDPA (built-in alternative)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        sdpa_available = True
        logger.info("✓ PyTorch SDPA available (built-in Flash Attention alternative)")
    
    return flash_attn_available, sdpa_available

FLASH_ATTN_AVAILABLE, SDPA_AVAILABLE = check_flash_attention()

# ============================================
# Print GPU Info
# ============================================
def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        gpu_id = 0
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        logger.info(f"Using GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check compute capability for Flash Attention
        props = torch.cuda.get_device_properties(gpu_id)
        compute_cap = f"{props.major}.{props.minor}"
        if props.major >= 8:
            logger.info(f"GPU Compute Capability: {compute_cap} (Flash Attention compatible)")
        else:
            logger.info(f"GPU Compute Capability: {compute_cap} (Flash Attention requires SM 8.0+)")
    else:
        logger.warning("No CUDA GPU available!")

print_gpu_info()

# ============================================
# Model Configurations
# ============================================
# Models are stored in HuggingFace cache format
# We'll use HF model IDs with custom cache_dir
HF_CACHE_DIR = r"D:\AI_MODELS"

MODEL_CONFIGS = {
    "mistral": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "type": "text",
        "supports_vision": False,
    },
    "qwen-vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "type": "vision",
        "supports_vision": True,
    },
}

# ============================================
# Request/Response Models
# ============================================
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]  # Can be string or list for vision models

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict

# ============================================
# Global Model Storage
# ============================================
model = None
processor = None
tokenizer = None
model_name = None

# ============================================
# Load Model with Flash Attention
# ============================================
def load_model(model_key: str):
    """Load model with Flash Attention 2 optimization"""
    global model, processor, tokenizer, model_name
    
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}")
    
    config = MODEL_CONFIGS[model_key]
    model_id = config["model_id"]
    model_name = model_key
    
    logger.info(f"Loading {model_key} ({model_id})")
    logger.info(f"Cache directory: {HF_CACHE_DIR}")
    logger.info(f"Flash Attention 2: {'Enabled' if FLASH_ATTN_AVAILABLE else 'Disabled (using SDPA)'}")
    
    # Determine attention implementation
    if FLASH_ATTN_AVAILABLE:
        attn_implementation = "flash_attention_2"
    elif SDPA_AVAILABLE:
        attn_implementation = "sdpa"
    else:
        attn_implementation = "eager"
    
    logger.info(f"Attention Implementation: {attn_implementation}")
    
    if config["type"] == "vision":
        # Load Qwen-VL
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,  # Use cached files only
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
        )
        tokenizer = processor.tokenizer
        logger.info(f"✓ Qwen-VL loaded with {attn_implementation}")
        
    else:
        # Load Mistral
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"✓ Mistral loaded with {attn_implementation}")
    
    # Print memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {memory_used:.2f} GB / {memory_total:.1f} GB")
    
    return model

# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="Multi-Model AI Server v3 (Flash Attention)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model": model_name,
        "flash_attention": FLASH_ATTN_AVAILABLE,
        "sdpa": SDPA_AVAILABLE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint - OpenAI compatible"""
    global model, processor, tokenizer
    
    try:
        start_time = time.time()
        config = MODEL_CONFIGS.get(model_name, {})
        
        if config.get("supports_vision"):
            response_text, token_count = await generate_vision_response(request)
        else:
            response_text, token_count = await generate_text_response(request)
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        logger.info(f"Generated {token_count} tokens in {generation_time:.2f}s ({token_count/generation_time:.1f} tok/s)")
        
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=model_name,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": token_count // 2,
                "completion_tokens": token_count // 2,
                "total_tokens": token_count
            }
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_text_response(request: ChatRequest):
    """Generate response for text-only models (Mistral)"""
    # Build conversation
    messages = []
    for msg in request.messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        messages.append({"role": msg.role, "content": content})
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with optimized settings
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,  # Enable KV cache for faster generation
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    token_count = outputs.shape[1]
    
    return response, token_count

async def generate_vision_response(request: ChatRequest):
    """Generate response for vision models (Qwen-VL)"""
    from qwen_vl_utils import process_vision_info
    
    # Build messages with image support
    messages = []
    for msg in request.messages:
        if isinstance(msg.content, list):
            # Multi-modal content
            content_parts = []
            for part in msg.content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        content_parts.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:"):
                            # Base64 image
                            content_parts.append({"type": "image", "image": image_url})
                        else:
                            content_parts.append({"type": "image", "image": image_url})
            messages.append({"role": msg.role, "content": content_parts})
        else:
            messages.append({"role": msg.role, "content": msg.content})
    
    # Process with Qwen-VL
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate with optimized settings
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            use_cache=True,  # Enable KV cache
        )
    
    # Decode
    generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    token_count = outputs.shape[1]
    
    return response, token_count

# ============================================
# Main Entry Point
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model AI Server v3")
    parser.add_argument("--model", type=str, required=True, choices=["mistral", "qwen-vl"])
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    # Load model
    load_model(args.model)
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
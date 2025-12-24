"""
Qwen2.5-VL-7B Server with SDPA Optimization
============================================
Vision-Language model server for image analysis and text chat
Usage: python server_qwen.py --port 8002

Recommended GPU: RTX 4080 (16GB) or better
Start command:
    set CUDA_VISIBLE_DEVICES=0
    python server_qwen.py --port 8002
"""

import argparse
import base64
import io
import logging
import os
import time
from typing import List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
HF_CACHE_DIR = r"D:\AI_MODELS"
MODEL_NAME = "qwen-vl"

# ============================================
# Check Flash Attention / SDPA
# ============================================
def check_attention_backend():
    """Check available attention implementations"""
    flash_attn_available = False
    sdpa_available = False
    
    try:
        import flash_attn
        flash_attn_available = True
        logger.info(f"✓ Flash Attention 2 available (v{flash_attn.__version__})")
    except ImportError:
        logger.info("✗ flash-attn not installed")
    
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        sdpa_available = True
        logger.info("✓ PyTorch SDPA available (built-in)")
    
    return flash_attn_available, sdpa_available

FLASH_ATTN_AVAILABLE, SDPA_AVAILABLE = check_attention_backend()

# ============================================
# Print GPU Info
# ============================================
def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        gpu_id = 0
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        compute_cap = torch.cuda.get_device_properties(gpu_id).major
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"Compute Capability: {compute_cap}.x {'(Flash Attn OK)' if compute_cap >= 8 else ''}")
    else:
        logger.warning("No CUDA GPU available!")

print_gpu_info()

# ============================================
# Request/Response Models
# ============================================
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]

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

# ============================================
# Load Model
# ============================================
def load_model():
    """Load Qwen-VL with optimized attention"""
    global model, processor
    
    logger.info(f"Loading {MODEL_NAME} ({MODEL_ID})")
    logger.info(f"Cache: {HF_CACHE_DIR}")
    
    # Select attention implementation
    if FLASH_ATTN_AVAILABLE:
        attn_impl = "flash_attention_2"
    elif SDPA_AVAILABLE:
        attn_impl = "sdpa"
    else:
        attn_impl = "eager"
    
    logger.info(f"Attention: {attn_impl}")
    
    from transformers import AutoProcessor
    
    # Try to load with the correct model class for Qwen2.5-VL
    try:
        # First try Qwen2.5-VL specific class
        from transformers import Qwen2_5_VLForConditionalGeneration
        logger.info("Using Qwen2_5_VLForConditionalGeneration")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
        )
    except ImportError:
        # Fallback to AutoModelForVision2Seq
        from transformers import AutoModelForVision2Seq
        logger.info("Using AutoModelForVision2Seq")
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
        )
    
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )
    
    # Print memory usage
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✓ Qwen-VL loaded ({attn_impl})")
        logger.info(f"GPU Memory: {mem_used:.2f} GB / {mem_total:.1f} GB")

# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="Qwen-VL Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "sdpa": SDPA_AVAILABLE,
        "flash_attn": FLASH_ATTN_AVAILABLE,
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint - OpenAI compatible"""
    global model, processor
    
    try:
        start_time = time.time()
        
        # Import vision utils
        from qwen_vl_utils import process_vision_info
        
        # Build messages with image support
        messages = []
        for msg in request.messages:
            if isinstance(msg.content, list):
                content_parts = []
                for part in msg.content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_parts.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {}).get("url", "")
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
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                do_sample=request.temperature > 0,
                use_cache=True,
            )
        
        # Decode
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        token_count = outputs.shape[1]
        
        end_time = time.time()
        gen_time = end_time - start_time
        tokens_per_sec = token_count / gen_time if gen_time > 0 else 0
        
        logger.info(f"Generated {token_count} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
        return ChatResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=MODEL_NAME,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": generated_ids.shape[1],
                "total_tokens": token_count
            }
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-VL Server")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    load_model()
    
    logger.info(f"Starting Qwen-VL server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
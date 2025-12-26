"""
Mistral-7B-Instruct Server with SDPA Optimization
=================================================
Text generation model server
Usage: python server_mistral.py --port 8001

Recommended GPU: RTX 3080 (10GB) or better, can use multiple GPUs
Start command:
    set CUDA_VISIBLE_DEVICES=1,2,3,4
    python server_mistral.py --port 8001
"""

import argparse
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
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
HF_CACHE_DIR = r"D:\AI_MODELS"
MODEL_NAME = "mistral"

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
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
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
tokenizer = None

# ============================================
# Load Model
# ============================================
def load_model():
    """Load Mistral with optimized attention"""
    global model, tokenizer
    
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
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=attn_impl,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Print memory usage
    if torch.cuda.is_available():
        total_mem = 0
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            if mem > 0:
                total_mem += mem
                logger.info(f"GPU {i} Memory: {mem:.2f} GB used")
        logger.info(f"✓ Mistral loaded ({attn_impl})")
        logger.info(f"Total GPU Memory: {total_mem:.2f} GB")

# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="Mistral Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append(torch.cuda.get_device_name(i))
    
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "gpus": gpus,
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
    global model, tokenizer
    
    try:
        start_time = time.time()
        
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
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else None,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        
        # Decode
        response_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        output_length = outputs.shape[1] - input_length
        
        end_time = time.time()
        gen_time = end_time - start_time
        tokens_per_sec = output_length / gen_time if gen_time > 0 else 0
        
        logger.info(f"Generated {output_length} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
        
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
                "prompt_tokens": input_length,
                "completion_tokens": output_length,
                "total_tokens": outputs.shape[1]
            }
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral Server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    load_model()
    
    logger.info(f"Starting Mistral server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

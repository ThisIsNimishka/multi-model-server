"""
Windows-Compatible Multi-Model Server with Flash Attention
Uses Hugging Face Transformers with Flash Attention 2
"""

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import time
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Model AI Server (Flash Attention)", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================

MODELS_BASE = "D:/AI_MODELS"

MODEL_CONFIGS = {
    "mistral": {
        "path": f"{MODELS_BASE}/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a",
        "type": "causal",
    },
    "qwen-vl": {
        "path": f"{MODELS_BASE}/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "type": "qwen-vl",
    },
    "gemma": {
        "path": f"{MODELS_BASE}/cache/models--google--gemma-7b-it/snapshots/9c5798d27f588501ce1e108079d2a19e4c3a2353",
        "type": "causal",
    },
}

# Active model (set via command line or env)
ACTIVE_MODEL = os.environ.get("MODEL_NAME", "mistral")

# Global model and tokenizer
model = None
tokenizer = None
processor = None
USE_FLASH_ATTENTION = False

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class Message(BaseModel):
    role: str
    content: Any

class ChatRequest(BaseModel):
    model: str = "default"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 512
    stream: bool = False

class ChatChoice(BaseModel):
    index: int = 0
    message: Dict[str, str]
    finish_reason: str = "stop"

class ChatResponse(BaseModel):
    id: str = "chatcmpl-local"
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatChoice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# ============================================================
# FLASH ATTENTION CHECK
# ============================================================

def check_flash_attention():
    """Check if Flash Attention is available"""
    global USE_FLASH_ATTENTION
    try:
        import flash_attn
        USE_FLASH_ATTENTION = True
        logger.info(f"✓ Flash Attention {flash_attn.__version__} available!")
        return True
    except ImportError:
        logger.warning("✗ Flash Attention not installed. Using standard attention.")
        logger.warning("  Install with: pip install flash-attn --no-build-isolation")
        return False

# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_name: str):
    """Load the specified model with Flash Attention if available"""
    global model, tokenizer, processor, ACTIVE_MODEL
    
    ACTIVE_MODEL = model_name
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    
    # Check Flash Attention
    flash_available = check_flash_attention()
    
    logger.info(f"")
    logger.info(f"{'='*60}")
    logger.info(f"Loading: {model_name}")
    logger.info(f"Path: {model_path}")
    logger.info(f"Flash Attention: {'ENABLED' if flash_available else 'DISABLED'}")
    logger.info(f"GPUs Available: {torch.cuda.device_count()}")
    logger.info(f"{'='*60}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    # Common loading kwargs
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    # Add Flash Attention if available
    if flash_available:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    
    if config["type"] == "qwen-vl":
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = processor.tokenizer
            logger.info(f"✓ Loaded Qwen-VL model")
        except Exception as e:
            logger.error(f"Qwen-VL specific loading failed: {e}")
            logger.info("Falling back to standard AutoModel...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Log memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    logger.info(f"✓ Model loaded successfully!")

# ============================================================
# CHAT FORMATTING
# ============================================================

def format_chat_prompt(messages: List[Message], model_name: str) -> str:
    """Format messages into a prompt string based on model type"""
    
    if "mistral" in model_name.lower():
        prompt = ""
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if msg.role == "user":
                prompt += f"[INST] {content} [/INST]"
            elif msg.role == "assistant":
                prompt += f" {content}</s>"
            elif msg.role == "system":
                prompt = f"[INST] {content}\n\n"
        return prompt
    
    elif "qwen" in model_name.lower():
        prompt = ""
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if msg.role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif msg.role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif msg.role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
    
    elif "gemma" in model_name.lower():
        prompt = ""
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if msg.role == "user":
                prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif msg.role == "assistant":
                prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"
        return prompt
    
    else:
        prompt = ""
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            prompt += f"{msg.role}: {content}\n"
        prompt += "assistant: "
        return prompt

# ============================================================
# GENERATION
# ============================================================

def generate_response(messages: List[Message], temperature: float, max_tokens: int) -> tuple:
    """Generate a response from the model"""
    global model, tokenizer
    
    prompt = format_chat_prompt(messages, ACTIVE_MODEL)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    input_length = inputs["input_ids"].shape[1]
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generation_time = time.time() - start_time
    output_length = outputs.shape[1] - input_length
    tokens_per_second = output_length / generation_time if generation_time > 0 else 0
    
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip(), {
        "input_tokens": input_length,
        "output_tokens": output_length,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
    }

# ============================================================
# API ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup():
    """Load model on startup"""
    load_model(ACTIVE_MODEL)

@app.get("/health")
async def health():
    """Health check with detailed info"""
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(i) / 1024**3, 2),
            })
    
    return {
        "status": "healthy",
        "model": ACTIVE_MODEL,
        "flash_attention": USE_FLASH_ATTENTION,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpu_info,
    }

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {"id": ACTIVE_MODEL, "object": "model", "owned_by": "local"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        response_text, stats = generate_response(
            request.messages,
            request.temperature,
            request.max_tokens,
        )
        
        logger.info(f"Generated {stats['output_tokens']} tokens in {stats['generation_time']:.2f}s ({stats['tokens_per_second']:.1f} tok/s)")
        
        return ChatResponse(
            created=int(time.time()),
            model=ACTIVE_MODEL,
            choices=[
                ChatChoice(
                    message={"role": "assistant", "content": response_text}
                )
            ],
            usage={
                "prompt_tokens": stats["input_tokens"],
                "completion_tokens": stats["output_tokens"],
                "total_tokens": stats["input_tokens"] + stats["output_tokens"],
            }
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Multi-Model AI Server (Flash Attention)",
        "model": ACTIVE_MODEL,
        "flash_attention": USE_FLASH_ATTENTION,
        "endpoints": {
            "/v1/chat/completions": "Chat completions (OpenAI compatible)",
            "/v1/models": "List available models",
            "/health": "Health check",
        }
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Windows Multi-Model Server with Flash Attention")
    parser.add_argument("--model", type=str, default="mistral",
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to load")
    parser.add_argument("--port", type=int, default=8001,
                       help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    args = parser.parse_args()
    
    ACTIVE_MODEL = args.model
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║       Multi-Model AI Server (Flash Attention)                ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Model: {args.model:<51} ║
    ║  Port:  {args.port:<51} ║
    ║  GPUs:  {torch.cuda.device_count():<51} ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=args.host, port=args.port)

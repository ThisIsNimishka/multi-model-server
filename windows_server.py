"""
Windows-Compatible Multi-Model Server
Uses Hugging Face Transformers directly (no vLLM/uvloop dependency)
"""

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import time
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Model AI Server", version="1.0.0")

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
        "device_map": "auto",
    },
    "qwen-vl": {
        "path": f"{MODELS_BASE}/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "type": "qwen-vl",
        "device_map": "auto",
    },
    "gemma": {
        "path": f"{MODELS_BASE}/cache/models--google--gemma-7b-it/snapshots/9c5798d27f588501ce1e108079d2a19e4c3a2353",
        "type": "causal",
        "device_map": "auto",
    },
}

# Which model to load (change this or use command line arg)
ACTIVE_MODEL = os.environ.get("MODEL_NAME", "mistral")

# Global model and tokenizer
model = None
tokenizer = None
processor = None  # For vision models

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class Message(BaseModel):
    role: str
    content: Any  # Can be string or list (for vision)

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
# MODEL LOADING
# ============================================================

def load_model(model_name: str):
    """Load the specified model"""
    global model, tokenizer, processor
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    
    logger.info(f"Loading {model_name} from {model_path}...")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    if config["type"] == "qwen-vl":
        # Qwen-VL requires special loading
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = processor.tokenizer
            logger.info(f"Loaded Qwen-VL model")
        except Exception as e:
            logger.error(f"Failed to load Qwen-VL: {e}")
            logger.info("Falling back to standard loading...")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        # Standard causal model loading
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded successfully!")
    logger.info(f"Model device: {next(model.parameters()).device}")

# ============================================================
# CHAT FORMATTING
# ============================================================

def format_chat_prompt(messages: List[Message], model_name: str) -> str:
    """Format messages into a prompt string"""
    
    if "mistral" in model_name.lower():
        # Mistral format
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
        # Qwen format
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
        # Gemma format
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
        # Generic format
        prompt = ""
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            prompt += f"{msg.role}: {content}\n"
        prompt += "assistant: "
        return prompt

# ============================================================
# GENERATION
# ============================================================

def generate_response(messages: List[Message], temperature: float, max_tokens: int) -> str:
    """Generate a response from the model"""
    global model, tokenizer
    
    prompt = format_chat_prompt(messages, ACTIVE_MODEL)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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
    
    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()

# ============================================================
# API ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup():
    """Load model on startup"""
    load_model(ACTIVE_MODEL)

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model": ACTIVE_MODEL,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
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
    
    start_time = time.time()
    
    try:
        response_text = generate_response(
            request.messages,
            request.temperature,
            request.max_tokens,
        )
        
        latency = time.time() - start_time
        logger.info(f"Generated response in {latency:.2f}s")
        
        return ChatResponse(
            created=int(time.time()),
            model=ACTIVE_MODEL,
            choices=[
                ChatChoice(
                    message={"role": "assistant", "content": response_text}
                )
            ]
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Multi-Model AI Server (Windows)",
        "model": ACTIVE_MODEL,
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
    
    parser = argparse.ArgumentParser(description="Windows-Compatible Multi-Model Server")
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
    ╔══════════════════════════════════════════════════════════╗
    ║          Multi-Model AI Server (Windows)                 ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Model: {args.model:<47} ║
    ║  Port:  {args.port:<47} ║
    ║  GPUs:  {torch.cuda.device_count():<47} ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(app, host=args.host, port=args.port)

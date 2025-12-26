"""
============================================================
MODEL NAME Server
============================================================
Port: 800X | GPUs: X,X | VRAM: ~XXGB

To use this template:
1. Copy this file: servers/_template.py ? servers/your_model.py
2. Edit the CONFIG section below
3. Run: python servers/your_model.py
============================================================
"""

# ==============================================================================
# CONFIG - Edit these values for your model
# ==============================================================================

MODEL_NAME = "your-model-name"                    # Display name
MODEL_PATH = "D:/AI_MODELS/models--org--model/snapshots/abc123"  # Local path
PORT = 8003                                       # Server port
GPU_IDS = "4,5"                                   # Comma-separated GPU IDs
MAX_MODEL_LEN = 4096                              # Max context length
GPU_MEMORY_UTILIZATION = 0.85                     # 0.0 to 1.0
TENSOR_PARALLEL_SIZE = 2                          # Number of GPUs to use
TRUST_REMOTE_CODE = True                          # Required for some models
VISION_ENABLED = False                            # Set True for vision models

# ==============================================================================
# SERVER CODE - No need to modify below this line
# ==============================================================================

import os
import sys
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

# Set GPU visibility before importing torch/vllm
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(MODEL_NAME)

# ==============================================================================
# Request/Response Models
# ==============================================================================

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_NAME
    messages: List[Message]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage

# ==============================================================================
# Model Loading
# ==============================================================================

llm = None
tokenizer = None

def load_model():
    global llm, tokenizer
    
    logger.info(f"Loading {MODEL_NAME} from {MODEL_PATH}")
    logger.info(f"Using GPUs: {GPU_IDS}")
    
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=TRUST_REMOTE_CODE,
            dtype="float16",
        )
        
        logger.info(f"? {MODEL_NAME} loaded successfully on port {PORT}")
        return True
        
    except Exception as e:
        logger.error(f"? Failed to load model: {e}")
        return False

# ==============================================================================
# FastAPI App
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    yield
    # Shutdown
    logger.info(f"Shutting down {MODEL_NAME} server")

app = FastAPI(
    title=f"{MODEL_NAME} API",
    description=f"OpenAI-compatible API for {MODEL_NAME}",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# Endpoints
# ==============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy" if llm else "loading",
        "model": MODEL_NAME,
        "port": PORT,
        "gpus": GPU_IDS
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "local"
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    from vllm import SamplingParams
    import time
    import uuid
    
    # Build prompt
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"[INST] {msg.content} [/INST]\n"
        elif msg.role == "user":
            prompt += f"[INST] {msg.content} [/INST]\n"
        elif msg.role == "assistant":
            prompt += f"{msg.content}\n"
    
    # Generate
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    
    outputs = llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    # Response
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=MODEL_NAME,
        choices=[ChatChoice(
            index=0,
            message=Message(role="assistant", content=generated_text),
            finish_reason="stop"
        )],
        usage=Usage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(generated_text.split()),
            total_tokens=len(prompt.split()) + len(generated_text.split())
        )
    )

# ==============================================================================
# Run Server
# ==============================================================================

if __name__ == "__main__":
    print(f"""
    +----------------------------------------------------------+
    ¦  {MODEL_NAME.upper():^52}  ¦
    ¦  Port: {PORT}  |  GPUs: {GPU_IDS:^10}                        ¦
    +----------------------------------------------------------+
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )

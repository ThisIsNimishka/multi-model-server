"""
Unified API Router for Multi-Model Server
Provides a single endpoint that routes to different model backends

Features:
- OpenAI-compatible API
- Automatic model routing based on request
- Health checking and failover
- Request logging and metrics
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import asyncio
import json
import time
import logging
from typing import Dict, Optional, List, AsyncGenerator
from pydantic import BaseModel
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Model API Router",
    description="Unified API for multiple LLM backends",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class BackendConfig:
    url: str
    model_aliases: List[str]
    capabilities: List[str]  # "chat", "vision", "embeddings"
    priority: int = 1
    max_concurrent: int = 10
    timeout: float = 300.0

from dataclasses import dataclass

BACKENDS: Dict[str, BackendConfig] = {
    "qwen-vl": BackendConfig(
        url="http://localhost:8001",
        model_aliases=["qwen", "qwen-vl", "qwen2.5-vl-7b", "vision"],
        capabilities=["chat", "vision"],
        priority=1,
    ),
    "mistral": BackendConfig(
        url="http://localhost:8002",
        model_aliases=["mistral", "mistral-7b", "default"],
        capabilities=["chat"],
        priority=1,
    ),
    "gemma": BackendConfig(
        url="http://localhost:8003",
        model_aliases=["gemma", "gemma-7b"],
        capabilities=["chat"],
        priority=2,
    ),
    "qwen-72b": BackendConfig(
        url="http://localhost:8004",
        model_aliases=["qwen-72b", "qwen2.5-vl-72b", "large"],
        capabilities=["chat", "vision"],
        priority=3,
    ),
}

# Default backend for unmatched requests
DEFAULT_BACKEND = "mistral"

# ============================================================
# METRICS AND STATE
# ============================================================

class Metrics:
    def __init__(self):
        self.requests_total = defaultdict(int)
        self.requests_success = defaultdict(int)
        self.requests_failed = defaultdict(int)
        self.latency_sum = defaultdict(float)
        self.active_requests = defaultdict(int)
        self.last_health_check = {}
        self.backend_healthy = {}
        
metrics = Metrics()

# ============================================================
# HEALTH CHECKING
# ============================================================

async def check_backend_health(name: str, config: BackendConfig) -> bool:
    """Check if a backend is healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{config.url}/health")
            healthy = response.status_code == 200
            metrics.backend_healthy[name] = healthy
            metrics.last_health_check[name] = datetime.now().isoformat()
            return healthy
    except Exception as e:
        logger.warning(f"Health check failed for {name}: {e}")
        metrics.backend_healthy[name] = False
        return False

async def health_check_loop():
    """Periodically check all backends"""
    while True:
        for name, config in BACKENDS.items():
            await check_backend_health(name, config)
        await asyncio.sleep(30)

@app.on_event("startup")
async def startup():
    """Start background tasks"""
    asyncio.create_task(health_check_loop())
    # Initial health check
    for name, config in BACKENDS.items():
        await check_backend_health(name, config)

# ============================================================
# ROUTING LOGIC
# ============================================================

def find_backend(model: str, needs_vision: bool = False) -> Optional[str]:
    """Find the appropriate backend for a model request"""
    model_lower = model.lower() if model else ""
    
    # Check for exact or alias match
    for name, config in BACKENDS.items():
        if any(alias in model_lower for alias in config.model_aliases):
            if metrics.backend_healthy.get(name, False):
                return name
    
    # If vision is needed, find a vision-capable backend
    if needs_vision:
        for name, config in BACKENDS.items():
            if "vision" in config.capabilities and metrics.backend_healthy.get(name, False):
                return name
    
    # Fall back to default if healthy
    if metrics.backend_healthy.get(DEFAULT_BACKEND, False):
        return DEFAULT_BACKEND
    
    # Find any healthy backend
    for name in BACKENDS:
        if metrics.backend_healthy.get(name, False):
            return name
    
    return None

def request_has_images(messages: List[dict]) -> bool:
    """Check if the request contains images"""
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
    return False

# ============================================================
# API ENDPOINTS
# ============================================================

class ChatRequest(BaseModel):
    model: str = "default"
    messages: List[dict]
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"

@app.get("/health")
async def health():
    """Health check endpoint"""
    healthy_count = sum(1 for v in metrics.backend_healthy.values() if v)
    return {
        "status": "healthy" if healthy_count > 0 else "degraded",
        "backends": metrics.backend_healthy,
        "healthy_count": healthy_count,
        "total_count": len(BACKENDS),
    }

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    models = []
    for name, config in BACKENDS.items():
        if metrics.backend_healthy.get(name, False):
            for alias in config.model_aliases:
                models.append(ModelInfo(
                    id=alias,
                    owned_by=name,
                ))
    return {"object": "list", "data": [m.dict() for m in models]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Chat completions endpoint (OpenAI compatible)"""
    start_time = time.time()
    
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    model = body.get("model", "default")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    
    # Check if request needs vision capabilities
    needs_vision = request_has_images(messages)
    
    # Find appropriate backend
    backend_name = find_backend(model, needs_vision)
    if not backend_name:
        raise HTTPException(
            status_code=503, 
            detail="No healthy backends available"
        )
    
    backend = BACKENDS[backend_name]
    metrics.requests_total[backend_name] += 1
    metrics.active_requests[backend_name] += 1
    
    logger.info(f"Routing request to {backend_name} (model: {model}, vision: {needs_vision})")
    
    try:
        async with httpx.AsyncClient(timeout=backend.timeout) as client:
            if stream:
                # Streaming response
                async def generate() -> AsyncGenerator[bytes, None]:
                    async with client.stream(
                        "POST",
                        f"{backend.url}/v1/chat/completions",
                        json=body,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                )
            else:
                # Non-streaming response
                response = await client.post(
                    f"{backend.url}/v1/chat/completions",
                    json=body,
                )
                
                latency = time.time() - start_time
                metrics.latency_sum[backend_name] += latency
                metrics.requests_success[backend_name] += 1
                
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code,
                )
                
    except httpx.TimeoutException:
        metrics.requests_failed[backend_name] += 1
        raise HTTPException(status_code=504, detail="Backend timeout")
    except httpx.ConnectError:
        metrics.requests_failed[backend_name] += 1
        metrics.backend_healthy[backend_name] = False
        raise HTTPException(status_code=503, detail=f"Backend {backend_name} unavailable")
    except Exception as e:
        metrics.requests_failed[backend_name] += 1
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        metrics.active_requests[backend_name] -= 1

@app.get("/metrics")
async def get_metrics():
    """Get server metrics"""
    result = {}
    for name in BACKENDS:
        total = metrics.requests_total[name]
        result[name] = {
            "requests_total": total,
            "requests_success": metrics.requests_success[name],
            "requests_failed": metrics.requests_failed[name],
            "active_requests": metrics.active_requests[name],
            "avg_latency": metrics.latency_sum[name] / total if total > 0 else 0,
            "healthy": metrics.backend_healthy.get(name, False),
            "last_health_check": metrics.last_health_check.get(name),
        }
    return result

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Multi-Model API Router",
        "version": "1.0.0",
        "endpoints": {
            "/v1/chat/completions": "Chat completions (OpenAI compatible)",
            "/v1/models": "List available models",
            "/health": "Health check",
            "/metrics": "Server metrics",
        },
        "backends": list(BACKENDS.keys()),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_router:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

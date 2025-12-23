import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Model AI Server", version="1.0.0")

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

MODELS_BASE = "D:/AI_MODELS"

MODEL_CONFIGS = {
    "mistral": {
        "path": f"{MODELS_BASE}/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a",
        "type": "causal",
    },
    "qwen-vl": {
        "path": f"{MODELS_BASE}/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "type": "qwen2.5-vl",  # Changed from qwen-vl
    },
    "gemma": {
        "path": f"{MODELS_BASE}/cache/models--google--gemma-7b-it/snapshots/9c5798d27f588501ce1e108079d2a19e4c3a2353",
        "type": "causal",
    },
}

ACTIVE_MODEL = os.environ.get("MODEL_NAME", "mistral")
model = None
tokenizer = None
processor = None
USE_FLASH_ATTENTION = False


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


def check_flash_attention():
    global USE_FLASH_ATTENTION
    try:
        import flash_attn
        USE_FLASH_ATTENTION = True
        logger.info(f"Flash Attention {flash_attn.__version__} available!")
        return True
    except ImportError:
        logger.warning("Flash Attention not installed. Using standard attention.")
        return False


def load_model(model_name: str):
    global model, tokenizer, processor, ACTIVE_MODEL
    
    ACTIVE_MODEL = model_name
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    model_type = config["type"]
    
    flash_available = check_flash_attention()
    
    logger.info("=" * 60)
    logger.info(f"Loading: {model_name}")
    logger.info(f"Type: {model_type}")
    logger.info(f"Path: {model_path}")
    logger.info(f"Flash Attention: {'ENABLED' if flash_available else 'DISABLED'}")
    logger.info(f"GPUs: {torch.cuda.device_count()}")
    logger.info("=" * 60)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    load_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    if flash_available:
        load_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Handle different model types
    if model_type == "qwen2.5-vl":
        # Qwen2.5-VL (Vision-Language) - newer version
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            logger.info("Loading Qwen2.5-VL model...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = processor.tokenizer
            logger.info("Qwen2.5-VL loaded successfully!")
            
        except ImportError as e:
            logger.error(f"Qwen2.5-VL not available in transformers: {e}")
            logger.info("Please upgrade transformers: pip install -U transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL: {e}")
            raise
            
    elif model_type == "qwen-vl":
        # Older Qwen-VL
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = processor.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load Qwen-VL: {e}")
            raise
    else:
        # Standard causal LM (Mistral, Gemma, etc.)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            logger.info(f"GPU {i}: {allocated:.2f}GB allocated")
    
    logger.info("Model loaded successfully!")


def format_chat_prompt(messages: List[Message], model_name: str) -> str:
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


def generate_response(messages: List[Message], temperature: float, max_tokens: int) -> tuple:
    global model, tokenizer, processor, ACTIVE_MODEL
    
    config = MODEL_CONFIGS.get(ACTIVE_MODEL, {})
    model_type = config.get("type", "causal")
    
    # For vision-language models, use processor
    if model_type in ["qwen2.5-vl", "qwen-vl"] and processor is not None:
        # Build conversation for Qwen VL
        conversation = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conversation.append({"role": msg.role, "content": content})
        
        # Apply chat template
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        # Standard text models
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


@app.on_event("startup")
async def startup():
    load_model(ACTIVE_MODEL)


@app.get("/health")
async def health():
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": round(torch.cuda.memory_allocated(i) / 1024**3, 2),
            })
    
    return {
        "status": "healthy",
        "model": ACTIVE_MODEL,
        "flash_attention": USE_FLASH_ATTENTION,
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpu_info,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": ACTIVE_MODEL, "object": "model", "owned_by": "local"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
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
            choices=[ChatChoice(message={"role": "assistant", "content": response_text})],
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
    return {
        "name": "Multi-Model AI Server",
        "model": ACTIVE_MODEL,
        "flash_attention": USE_FLASH_ATTENTION,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistral", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    ACTIVE_MODEL = args.model
    
    print(f"\n{'='*60}")
    print(f"  Multi-Model AI Server")
    print(f"  Model: {args.model}")
    print(f"  Port:  {args.port}")
    print(f"  GPUs:  {torch.cuda.device_count()}")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=args.host, port=args.port)
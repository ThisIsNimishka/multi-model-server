# 🚀 Multi-Model AI Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)

Serve multiple LLM models simultaneously across multiple GPUs with an OpenAI-compatible API.

---

## ✨ Features

- **Multi-Model** — Run different models on different GPUs
- **OpenAI-Compatible** — Drop-in replacement API
- **Vision Support** — Image understanding with VL models
- **SDPA Optimized** — Fast inference with PyTorch's scaled dot-product attention
- **Template System** — Add new models in minutes
- **Modern Web UI** — Glass-morphism chat interface

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **API** | FastAPI + Uvicorn |
| **Models** | HuggingFace Transformers |
| **Backend** | PyTorch + CUDA |
| **Optimization** | SDPA / Flash Attention 2 |

---

## 📁 Structure

```
multi-model-server/
├── frontend/           # Web interface
├── servers/            # Model servers
│   ├── _template.py    # ⭐ Copy for new models
│   ├── mistral.py      # Mistral-7B
│   ├── qwen.py         # Qwen-VL (vision)
│   └── router.py       # API router
├── scripts/            # Batch scripts
└── tests/
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/multi-model-server.git
cd multi-model-server
pip install -r requirements.txt
```

### 2. Download a Model

```bash
pip install huggingface_hub
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2
```

### 3. Configure

Edit `servers/mistral.py` CONFIG section:

```python
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "mistral"
HF_CACHE_DIR = None          # Your cache dir or None for default
DEFAULT_PORT = 8001
LOCAL_FILES_ONLY = False     # True if already downloaded
```

### 4. Run

```bash
# Single GPU
python servers/mistral.py --port 8001

# Specific GPUs
set CUDA_VISIBLE_DEVICES=0,1
python servers/mistral.py --port 8001
```

---

## 🆕 Adding a Model

```bash
# 1. Copy template
cp servers/_template.py servers/llama.py

# 2. Edit CONFIG section at top:
MODEL_ID = "meta-llama/Llama-3-8B-Instruct"
MODEL_NAME = "llama"
DEFAULT_PORT = 8003

# 3. Run
python servers/llama.py --port 8003
```

---

## 📡 API

### Chat Completion

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral","messages":[{"role":"user","content":"Hello!"}]}'
```

### Python

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8001/v1", api_key="x")
response = client.chat.completions.create(
    model="mistral",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |

---

## ⚙️ Config Options

| Option | Description |
|--------|-------------|
| `MODEL_ID` | HuggingFace model ID |
| `MODEL_NAME` | Display name for API |
| `HF_CACHE_DIR` | Cache directory (None = default) |
| `DEFAULT_PORT` | Server port |
| `LOCAL_FILES_ONLY` | Skip download, use cached |

---

## 🐛 Troubleshooting

**OOM Error**
```bash
# Use fewer/smaller GPUs
set CUDA_VISIBLE_DEVICES=0
```

**Port in use**
```bash
# Windows
netstat -ano | findstr :8001
taskkill /PID <pid> /F
```

**Model not found**
```bash
# Download first
huggingface-cli download <model-id>
```

---

## 📄 License

MIT

---

<p align="center">Made with ❤️ for the AI community</p>
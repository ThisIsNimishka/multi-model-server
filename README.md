# 🚀 Multi-Model AI Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A production-ready system for serving multiple LLM models simultaneously across multiple GPUs with a beautiful web interface.

🌐 **Live Demo:** [thisisnimishka.github.io/multi-model-server](https://thisisnimishka.github.io/multi-model-server/)

---

## ✨ Features

- **Multi-Model Support** — Run multiple models simultaneously on different GPUs
- **OpenAI-Compatible API** — Drop-in replacement for OpenAI API
- **Vision Support** — Qwen-VL for image understanding
- **Beautiful Web UI** — Modern glass-morphism chat interface
- **Easy Model Addition** — Template-based system for adding new models
- **Health Monitoring** — Built-in health checks per model

---

## 📁 Project Structure

```
multi-model-server/
│
├── frontend/                 # Web UI (GitHub Pages)
│   ├── index.html            # Live chat interface
│   ├── chat_loading.html     # Working version
│   └── chat_server_down.html # Offline page
│
├── servers/                  # Model servers
│   ├── _template.py          # ⭐ Copy this for new models
│   ├── mistral.py            # Mistral-7B server
│   ├── qwen.py               # Qwen-VL server
│   ├── router.py             # API router
│   └── manager.py            # Server manager
│
├── scripts/                  # Startup scripts
│   ├── start.bat             # Start all servers
│   ├── stop.bat              # Stop all servers
│   └── install.bat           # Install dependencies
│
├── tests/                    # Test files
│   └── test_client.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x 10GB+ VRAM | Multi-GPU setup |
| RAM | 16GB | 32GB+ |
| Storage | 50GB SSD | 500GB+ SSD |
| CUDA | 12.0+ | 12.4+ |

### Current Setup (Example)
- 1x RTX 4080 (16GB) — Qwen-VL
- 7x RTX 3080 (10GB) — Mistral, Gemma, Llama, etc.

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/ThisIsNimishka/multi-model-server.git
cd multi-model-server
```

### 2. Install Dependencies

```bash
# Windows
scripts\install.bat

# Manual
pip install -r requirements.txt
```

### 3. Download Models

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
```

### 4. Configure & Run

Edit the CONFIG section in `servers/mistral.py`:

```python
MODEL_PATH = "D:/AI_MODELS/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/..."
PORT = 8001
GPU_IDS = "0,1"
```

Run the server:

```bash
python servers/mistral.py
```

---

## 🆕 Adding a New Model

### Step 1: Copy Template

```bash
copy servers\_template.py servers\llama.py
```

### Step 2: Edit CONFIG Section

Open `servers/llama.py` and edit only the top section:

```python
# === CONFIG ===
MODEL_NAME = "llama-3-8b"
MODEL_PATH = "D:/AI_MODELS/models--meta-llama--Llama-3-8B/snapshots/..."
PORT = 8003
GPU_IDS = "4,5"
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.85
TENSOR_PARALLEL_SIZE = 2
```

### Step 3: Run

```bash
python servers/llama.py
```

**That's it!** Your new model is now serving on the specified port.

---

## 📡 API Usage

### OpenAI-Compatible Endpoint

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="mistral",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### cURL

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Health Check

```bash
curl http://localhost:8001/health
```

---

## 🔌 Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |

---

## 🖥️ Model Servers

| Model | Port | GPUs | Description |
|-------|------|------|-------------|
| Mistral-7B | 8001 | 2,3 | Fast text generation |
| Qwen-VL | 8002 | 0,1 | Vision + Language |
| *Add more...* | 800X | X,X | Use `_template.py` |

---

## 🌐 Web Interface

The frontend is hosted on GitHub Pages:

**Live:** [thisisnimishka.github.io/multi-model-server](https://thisisnimishka.github.io/multi-model-server/)

### Features
- Modern glass-morphism design
- Model selection cards
- Real-time chat with streaming
- Image upload for vision models
- Token/speed statistics

### Switching Pages

| File | Purpose |
|------|---------|
| `chat_loading.html` | Normal working interface |
| `chat_server_down.html` | Offline/maintenance page |

To switch: Copy content to `index.html` and push.

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```python
# Reduce in CONFIG section:
GPU_MEMORY_UTILIZATION = 0.80
MAX_MODEL_LEN = 2048
```

### Port Already in Use

```bash
# Windows - find process
netstat -ano | findstr :8001

# Kill process
taskkill /PID <pid> /F
```

### Model Loading Fails

```bash
# Verify model path exists
dir D:\AI_MODELS\models--mistralai--Mistral-7B-Instruct-v0.2\snapshots\
```

---

## 📋 Scripts

| Script | Description |
|--------|-------------|
| `scripts/start.bat` | Start all model servers |
| `scripts/stop.bat` | Stop all model servers |
| `scripts/install.bat` | Install Python dependencies |

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) — High-throughput LLM serving
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [HuggingFace](https://huggingface.co/) — Model hosting

---

<p align="center">
  Made with ❤️ for the AI community
</p>
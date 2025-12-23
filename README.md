# 🚀 Multi-Model AI Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![vLLM](https://img.shields.io/badge/vLLM-0.6+-orange.svg)](https://github.com/vllm-project/vllm)

A production-ready system for serving multiple LLM models simultaneously across multiple GPUs. Run Qwen, Mistral, Gemma, and more with a unified OpenAI-compatible API.

## ✨ Features

- **Multi-Model Support** - Run multiple models simultaneously on different GPUs
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API
- **Smart Routing** - Automatic routing based on model name or capabilities
- **Vision Support** - Qwen-VL for image understanding
- **Health Monitoring** - Built-in health checks and metrics
- **Easy Configuration** - Simple batch scripts for Windows
- **Streaming** - Full streaming response support
- **Load Balancing** - Distribute requests across healthy backends

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Configuration](#-configuration)
- [Supported Models](#-supported-models)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🏗 Architecture

```
                    ┌─────────────────────────────┐
                    │      API Router (:8000)      │
                    │   Unified OpenAI-compatible  │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────┬───────────┼───────────┬─────────────┐
         ▼             ▼           ▼           ▼             ▼
   ┌──────────┐  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Qwen-VL  │  │ Mistral  │ │  Gemma   │ │ Qwen-72B │ │ Flan-T5  │
   │  :8001   │  │  :8002   │ │  :8003   │ │  :8004   │ │  :8005   │
   │ GPU 0,1  │  │ GPU 2,3  │ │ GPU 4,5  │ │ GPU 0-7  │ │  GPU 6   │
   └──────────┘  └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## 💻 Hardware Requirements

| Component | Minimum | Your Setup |
|-----------|---------|------------|
| GPUs | 2x with 10GB+ | 1x RTX 4080 (16GB) + 7x RTX 3080 (10GB) |
| VRAM Total | 20GB | ~86GB |
| RAM | 32GB | - |
| Storage | 100GB SSD | 1TB SSD |
| CUDA | 12.0+ | 12.8 |

## 📦 Installation

### Option 1: Automated Installation (Windows)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/multi-model-server.git
cd multi-model-server

# Run the installation script
install.bat
```

### Option 2: Manual Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using pip (coming soon)

```bash
pip install multi-model-server
```

## 🚀 Quick Start

### 1. Download Models

Download your models using HuggingFace CLI or manually:

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2
```

### 2. Configure Model Paths

Edit `start_servers.bat` or `server_manager.py` with your model paths:

```batch
set MODELS_BASE=D:\AI_MODELS
set QWEN_PATH=%MODELS_BASE%\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\...
```

### 3. Start Servers

```bash
# Start default configuration (Qwen + Mistral + Router)
start_servers.bat

# Or start specific models
start_servers.bat qwen      # Only Qwen-VL
start_servers.bat mistral   # Only Mistral
start_servers.bat all       # All models
```

### 3. Test

```bash
# Run automated tests
python test_client.py test

# Interactive chat
python test_client.py chat
```

## 📡 API Usage

### OpenAI-Compatible Endpoint

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Chat completion
response = client.chat.completions.create(
    model="mistral",  # or "qwen", "gemma"
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Direct HTTP Requests

```bash
# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models
```

### Vision (Qwen-VL)

```python
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="qwen",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            }
        ]
    }]
)
```

## ⚙️ Configuration

### GPU Allocation

Edit the GPU assignments in `server_manager.py` or `start_servers.bat`:

| Model | Default GPUs | VRAM Required |
|-------|--------------|---------------|
| Qwen2.5-VL-7B | 0, 1 | ~20GB |
| Mistral-7B | 2, 3 | ~18GB |
| Gemma-7B | 4, 5 | ~18GB |
| Qwen-72B (INT4) | 0-7 | ~80GB |

### Model Paths

Update paths in `start_servers.bat`:

```batch
set QWEN_PATH=D:\AI_MODELS\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\...
set MISTRAL_PATH=D:\AI_MODELS\models--mistralai--Mistral-7B-Instruct-v0.2\snapshots\...
```

### Server Options

| Option | Description | Default |
|--------|-------------|---------|
| `--tensor-parallel-size` | Number of GPUs per model | 2 |
| `--max-model-len` | Maximum context length | 4096 |
| `--gpu-memory-utilization` | GPU memory fraction | 0.85 |
| `--dtype` | Data type (float16/bfloat16) | auto |

## 🔌 Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completion (OpenAI compatible) |
| `GET /v1/models` | List available models |
| `GET /health` | Health check |
| `GET /metrics` | Server metrics |

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.80

# Reduce context length
--max-model-len 2048
```

**2. Model loading fails**
```bash
# Check model path exists
dir D:\AI_MODELS\models--Qwen--Qwen2.5-VL-7B-Instruct\snapshots\

# Verify model files
python -c "from transformers import AutoModel; AutoModel.from_pretrained('path/to/model')"
```

**3. Port already in use**
```bash
# Find process using port
netstat -ano | findstr :8001

# Kill process
taskkill /PID <pid> /F
```

**4. vLLM installation issues on Windows**
```bash
# Try building from source
pip install vllm --no-binary vllm
```

### Logs

```bash
# View server logs
type server_manager.log

# Real-time monitoring
Get-Content server_manager.log -Wait
```

## Advanced Usage

### Running Qwen-72B (Requires All GPUs)

```batch
REM Stop other models first
REM Start 72B with all GPUs
set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m vllm.entrypoints.openai.api_server ^
    --model D:\AI_MODELS\models--Qwen--Qwen2.5-VL-72B-Instruct\snapshots\89c86200743eec961a297729e7990e8f2ddbc4c5 ^
    --port 8004 ^
    --tensor-parallel-size 8 ^
    --max-model-len 2048 ^
    --gpu-memory-utilization 0.95 ^
    --trust-remote-code
```

Note: 72B model may require AWQ/GPTQ quantization to fit in 86GB VRAM.

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  qwen:
    image: vllm/vllm-openai:latest
    ports:
      - "8001:8000"
    volumes:
      - D:/AI_MODELS:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']
              capabilities: [gpu]
    command: --model /models/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 --tensor-parallel-size 2
```

## 📁 Project Structure

```
multi-model-server/
├── server_manager.py    # Main server manager (Python)
├── api_router.py        # Unified API router with load balancing
├── start_servers.bat    # Windows startup script
├── stop_servers.bat     # Windows stop script
├── install.bat          # Installation script
├── test_client.py       # Test client and interactive chat
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License
└── .gitignore           # Git ignore rules
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/multi-model-server.git
cd multi-model-server

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python test_client.py test
```

### Areas for Contribution

- 🐧 Linux/Mac support improvements
- 🐳 Docker/Kubernetes deployment configs
- 📊 Prometheus metrics integration
- 🔐 Authentication/API key support
- 📚 Documentation improvements
- 🧪 Test coverage

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [HuggingFace](https://huggingface.co/) - Model hosting and transformers library
- Model creators: Qwen, Mistral AI, Google

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for the AI community
</p>

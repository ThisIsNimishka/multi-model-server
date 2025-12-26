"""
Multi-Model AI Server Manager
Manages multiple LLM servers across 8 GPUs

GPU Allocation Strategy:
- GPU 0 (4080-16GB) + GPU 1 (3080-10GB): Qwen2.5-VL-7B (Vision-Language)
- GPU 2 + GPU 3 (3080s): Mistral-7B-Instruct
- GPU 4 + GPU 5 (3080s): Gemma-7B (optional)
- GPU 6 + GPU 7 (3080s): Reserved or Qwen-72B-INT4
"""

import subprocess
import os
import sys
import time
import signal
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import threading
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SYSTEM
# ============================================================

MODELS_BASE = Path("D:/AI_MODELS")

# HuggingFace cache locations (your models are stored in multiple places)
HF_CACHE_LOCATIONS = [
    MODELS_BASE,
    MODELS_BASE / "cache" / "hub",
    MODELS_BASE / "cache",
    MODELS_BASE / "hub",
]

@dataclass
class ModelConfig:
    name: str
    model_id: str  # HuggingFace model ID or local path
    snapshot_path: str  # Relative path to snapshot folder
    gpus: str  # Comma-separated GPU IDs
    port: int
    tensor_parallel: int
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"  # auto, float16, bfloat16
    quantization: Optional[str] = None  # awq, gptq, squeezellm, None
    trust_remote_code: bool = True
    enabled: bool = True


# Model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "qwen-vl-7b": ModelConfig(
        name="Qwen2.5-VL-7B-Instruct",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        snapshot_path="models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5",
        gpus="0,1",
        port=8001,
        tensor_parallel=2,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enabled=True,
    ),
    "mistral-7b": ModelConfig(
        name="Mistral-7B-Instruct-v0.2",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        snapshot_path="models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a",
        gpus="2,3",
        port=8002,
        tensor_parallel=2,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enabled=True,
    ),
    "gemma-7b": ModelConfig(
        name="Gemma-7B-IT",
        model_id="google/gemma-7b-it",
        snapshot_path="cache/models--google--gemma-7b-it/snapshots/9c5798d27f588501ce1e108079d2a19e4c3a2353",
        gpus="4,5",
        port=8003,
        tensor_parallel=2,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        enabled=False,  # Disabled by default
    ),
    "qwen-72b-int4": ModelConfig(
        name="Qwen2.5-VL-72B-Instruct-INT4",
        model_id="Qwen/Qwen2.5-VL-72B-Instruct",
        snapshot_path="models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/89c86200743eec961a297729e7990e8f2ddbc4c5",
        gpus="0,1,2,3,4,5,6,7",  # All 8 GPUs
        port=8004,
        tensor_parallel=8,
        max_model_len=2048,  # Reduced for memory
        gpu_memory_utilization=0.95,
        quantization="awq",  # Requires AWQ quantized model
        enabled=False,  # Disabled - requires all GPUs
    ),
    "flan-t5": ModelConfig(
        name="Flan-T5-Base",
        model_id="google/flan-t5-base",
        snapshot_path="models--google--flan-t5-base/snapshots/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
        gpus="6",
        port=8005,
        tensor_parallel=1,
        max_model_len=512,
        gpu_memory_utilization=0.50,
        enabled=False,
    ),
}


class ModelServer:
    """Manages a single model server process"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.log_thread: Optional[threading.Thread] = None
        self._stop_logging = threading.Event()
        
    def find_model_path(self) -> Optional[Path]:
        """Find the model in various cache locations"""
        for base in HF_CACHE_LOCATIONS:
            full_path = base / self.config.snapshot_path
            if full_path.exists():
                logger.info(f"Found model at: {full_path}")
                return full_path
        
        # Try direct path
        direct_path = MODELS_BASE / self.config.snapshot_path
        if direct_path.exists():
            return direct_path
            
        return None
    
    def is_port_available(self) -> bool:
        """Check if the port is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', self.config.port)) != 0
    
    def build_command(self, model_path: Path) -> List[str]:
        """Build the vLLM server command"""
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(model_path),
            "--port", str(self.config.port),
            "--tensor-parallel-size", str(self.config.tensor_parallel),
            "--max-model-len", str(self.config.max_model_len),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--dtype", self.config.dtype,
        ]
        
        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")
            
        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])
            
        return cmd
    
    def _log_output(self, pipe, prefix: str):
        """Thread function to log subprocess output"""
        for line in iter(pipe.readline, b''):
            if self._stop_logging.is_set():
                break
            try:
                logger.info(f"[{prefix}] {line.decode().strip()}")
            except:
                pass
        pipe.close()
    
    def start(self) -> bool:
        """Start the model server"""
        if not self.config.enabled:
            logger.info(f"Model {self.config.name} is disabled, skipping")
            return False
            
        model_path = self.find_model_path()
        if not model_path:
            logger.error(f"Could not find model: {self.config.name}")
            logger.error(f"Looked in: {[str(p / self.config.snapshot_path) for p in HF_CACHE_LOCATIONS]}")
            return False
        
        if not self.is_port_available():
            logger.error(f"Port {self.config.port} is already in use")
            return False
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.gpus
        
        cmd = self.build_command(model_path)
        logger.info(f"Starting {self.config.name} on GPUs {self.config.gpus}, port {self.config.port}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
            )
            
            # Start logging threads
            self._stop_logging.clear()
            self.log_thread = threading.Thread(
                target=self._log_output,
                args=(self.process.stderr, self.config.name),
                daemon=True
            )
            self.log_thread.start()
            
            logger.info(f"Started {self.config.name} with PID {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {self.config.name}: {e}")
            return False
    
    def stop(self):
        """Stop the model server"""
        if self.process:
            logger.info(f"Stopping {self.config.name}...")
            self._stop_logging.set()
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            logger.info(f"Stopped {self.config.name}")
    
    def is_running(self) -> bool:
        """Check if the server is running"""
        return self.process is not None and self.process.poll() is None
    
    def health_check(self) -> bool:
        """Check if the server is responding"""
        import urllib.request
        try:
            url = f"http://localhost:{self.config.port}/health"
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status == 200
        except:
            return False


class ServerManager:
    """Manages multiple model servers"""
    
    def __init__(self):
        self.servers: Dict[str, ModelServer] = {}
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, stopping all servers...")
        self.stop_all()
        sys.exit(0)
    
    def add_model(self, name: str, config: ModelConfig):
        """Add a model to manage"""
        self.servers[name] = ModelServer(config)
        
    def start_model(self, name: str) -> bool:
        """Start a specific model"""
        if name not in self.servers:
            logger.error(f"Unknown model: {name}")
            return False
        return self.servers[name].start()
    
    def stop_model(self, name: str):
        """Stop a specific model"""
        if name in self.servers:
            self.servers[name].stop()
    
    def start_all(self, delay: int = 30):
        """Start all enabled models with delay between each"""
        enabled = [n for n, s in self.servers.items() if s.config.enabled]
        logger.info(f"Starting {len(enabled)} models: {enabled}")
        
        for i, name in enumerate(enabled):
            if self.start_model(name):
                if i < len(enabled) - 1:
                    logger.info(f"Waiting {delay}s before starting next model...")
                    time.sleep(delay)
    
    def stop_all(self):
        """Stop all running models"""
        for name, server in self.servers.items():
            if server.is_running():
                server.stop()
    
    def status(self) -> Dict[str, dict]:
        """Get status of all models"""
        result = {}
        for name, server in self.servers.items():
            result[name] = {
                "enabled": server.config.enabled,
                "running": server.is_running(),
                "port": server.config.port,
                "gpus": server.config.gpus,
                "healthy": server.health_check() if server.is_running() else False,
            }
        return result
    
    def print_status(self):
        """Print formatted status"""
        status = self.status()
        print("\n" + "="*70)
        print("MODEL SERVER STATUS")
        print("="*70)
        for name, info in status.items():
            status_icon = "✓" if info["running"] else "✗"
            health_icon = "♥" if info["healthy"] else "?"
            enabled = "enabled" if info["enabled"] else "disabled"
            print(f"{status_icon} {name:20} | Port: {info['port']:5} | GPUs: {info['gpus']:10} | {enabled:8} | Health: {health_icon}")
        print("="*70 + "\n")
    
    def monitor(self, interval: int = 60):
        """Monitor servers and restart if needed"""
        logger.info(f"Starting monitor (checking every {interval}s)")
        while True:
            for name, server in self.servers.items():
                if server.config.enabled and not server.is_running():
                    logger.warning(f"{name} is not running, attempting restart...")
                    server.start()
            time.sleep(interval)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Model AI Server Manager")
    parser.add_argument("command", choices=["start", "stop", "status", "monitor"],
                       help="Command to execute")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to start/stop (default: all enabled)")
    parser.add_argument("--delay", type=int, default=30,
                       help="Delay between starting models (seconds)")
    parser.add_argument("--enable", nargs="+",
                       help="Enable specific models before starting")
    args = parser.parse_args()
    
    # Create manager and add all models
    manager = ServerManager()
    for name, config in MODEL_CONFIGS.items():
        manager.add_model(name, config)
    
    # Enable specific models if requested
    if args.enable:
        for name in args.enable:
            if name in manager.servers:
                manager.servers[name].config.enabled = True
                logger.info(f"Enabled {name}")
    
    # Execute command
    if args.command == "start":
        if args.models:
            for name in args.models:
                manager.start_model(name)
                time.sleep(args.delay)
        else:
            manager.start_all(delay=args.delay)
        manager.print_status()
        
        # Keep running
        print("Servers running. Press Ctrl+C to stop all.")
        try:
            while True:
                time.sleep(60)
                manager.print_status()
        except KeyboardInterrupt:
            manager.stop_all()
            
    elif args.command == "stop":
        if args.models:
            for name in args.models:
                manager.stop_model(name)
        else:
            manager.stop_all()
            
    elif args.command == "status":
        manager.print_status()
        
    elif args.command == "monitor":
        manager.start_all(delay=args.delay)
        manager.monitor()


if __name__ == "__main__":
    main()

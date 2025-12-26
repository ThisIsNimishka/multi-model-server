"""
Test Client for Multi-Model Server
Tests all endpoints and models
"""

import httpx
import asyncio
import json
import time
from typing import Optional
import base64
from pathlib import Path

# Configuration
ROUTER_URL = "http://localhost:8000"
QWEN_URL = "http://localhost:8001"
MISTRAL_URL = "http://localhost:8002"
GEMMA_URL = "http://localhost:8003"

# Test prompts
TEST_PROMPTS = {
    "simple": "What is 2 + 2? Answer in one word.",
    "creative": "Write a haiku about artificial intelligence.",
    "code": "Write a Python function to calculate fibonacci numbers.",
    "reasoning": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
}


def print_header(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_result(success: bool, message: str, latency: Optional[float] = None):
    icon = "✓" if success else "✗"
    latency_str = f" ({latency:.2f}s)" if latency else ""
    print(f"  {icon} {message}{latency_str}")


async def check_health(client: httpx.AsyncClient, url: str, name: str) -> bool:
    """Check if a server is healthy"""
    try:
        response = await client.get(f"{url}/health", timeout=5.0)
        healthy = response.status_code == 200
        print_result(healthy, f"{name} health check")
        return healthy
    except Exception as e:
        print_result(False, f"{name} health check - {e}")
        return False


async def test_chat_completion(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    name: str
) -> bool:
    """Test a chat completion request"""
    start = time.time()
    try:
        response = await client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.7,
            },
            timeout=60.0,
        )
        latency = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print_result(True, f"{name}: Got response", latency)
            print(f"      Response: {content[:100]}...")
            return True
        else:
            print_result(False, f"{name}: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"{name}: {e}")
        return False


async def test_streaming(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str
) -> bool:
    """Test streaming response"""
    start = time.time()
    try:
        async with client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 50,
                "stream": True,
            },
            timeout=60.0,
        ) as response:
            chunks = 0
            async for chunk in response.aiter_bytes():
                chunks += 1
            
            latency = time.time() - start
            print_result(True, f"Streaming test: {chunks} chunks received", latency)
            return True
            
    except Exception as e:
        print_result(False, f"Streaming test: {e}")
        return False


async def test_vision(
    client: httpx.AsyncClient,
    url: str,
    model: str
) -> bool:
    """Test vision capability with a test image"""
    # Create a simple test image (1x1 red pixel)
    # In real use, you would load an actual image
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    
    try:
        response = await client.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{test_image_b64}"
                            }
                        }
                    ]
                }],
                "max_tokens": 50,
            },
            timeout=60.0,
        )
        
        if response.status_code == 200:
            print_result(True, "Vision test: Model accepted image input")
            return True
        else:
            print_result(False, f"Vision test: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print_result(False, f"Vision test: {e}")
        return False


async def test_list_models(client: httpx.AsyncClient, url: str) -> bool:
    """Test listing models"""
    try:
        response = await client.get(f"{url}/v1/models", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            print_result(True, f"List models: {models}")
            return True
        else:
            print_result(False, f"List models: HTTP {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"List models: {e}")
        return False


async def run_tests():
    """Run all tests"""
    print_header("Multi-Model Server Test Suite")
    
    async with httpx.AsyncClient() as client:
        # Test 1: Health checks
        print_header("1. Health Checks")
        router_healthy = await check_health(client, ROUTER_URL, "Router")
        qwen_healthy = await check_health(client, QWEN_URL, "Qwen-VL")
        mistral_healthy = await check_health(client, MISTRAL_URL, "Mistral")
        
        # Test 2: List models
        print_header("2. List Available Models")
        if router_healthy:
            await test_list_models(client, ROUTER_URL)
        
        # Test 3: Direct backend tests
        print_header("3. Direct Backend Tests")
        
        if qwen_healthy:
            await test_chat_completion(
                client, QWEN_URL, "qwen",
                TEST_PROMPTS["simple"], "Qwen-VL direct"
            )
        
        if mistral_healthy:
            await test_chat_completion(
                client, MISTRAL_URL, "mistral",
                TEST_PROMPTS["simple"], "Mistral direct"
            )
        
        # Test 4: Router routing tests
        print_header("4. Router Routing Tests")
        
        if router_healthy:
            # Test routing to different models
            await test_chat_completion(
                client, ROUTER_URL, "qwen",
                TEST_PROMPTS["simple"], "Router -> Qwen"
            )
            await test_chat_completion(
                client, ROUTER_URL, "mistral",
                TEST_PROMPTS["simple"], "Router -> Mistral"
            )
            await test_chat_completion(
                client, ROUTER_URL, "default",
                TEST_PROMPTS["simple"], "Router -> Default"
            )
        
        # Test 5: Streaming
        print_header("5. Streaming Tests")
        if mistral_healthy:
            await test_streaming(
                client, MISTRAL_URL, "mistral",
                TEST_PROMPTS["creative"]
            )
        
        # Test 6: Vision (Qwen-VL)
        print_header("6. Vision Tests")
        if qwen_healthy:
            await test_vision(client, QWEN_URL, "qwen")
        
        # Test 7: Different prompt types
        print_header("7. Prompt Type Tests")
        if mistral_healthy:
            for prompt_type, prompt in TEST_PROMPTS.items():
                await test_chat_completion(
                    client, MISTRAL_URL, "mistral",
                    prompt, f"Mistral - {prompt_type}"
                )
        
        # Summary
        print_header("Test Summary")
        print(f"  Router: {'✓ Healthy' if router_healthy else '✗ Down'}")
        print(f"  Qwen-VL: {'✓ Healthy' if qwen_healthy else '✗ Down'}")
        print(f"  Mistral: {'✓ Healthy' if mistral_healthy else '✗ Down'}")


async def interactive_chat():
    """Interactive chat mode"""
    print_header("Interactive Chat Mode")
    print("Type 'quit' to exit, 'switch <model>' to change models")
    print("Available models: qwen, mistral, gemma")
    print()
    
    current_model = "mistral"
    messages = []
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                user_input = input(f"[{current_model}] You: ").strip()
                
                if user_input.lower() == "quit":
                    break
                elif user_input.lower().startswith("switch "):
                    current_model = user_input.split()[1]
                    print(f"Switched to {current_model}")
                    continue
                elif user_input.lower() == "clear":
                    messages = []
                    print("Conversation cleared")
                    continue
                elif not user_input:
                    continue
                
                messages.append({"role": "user", "content": user_input})
                
                response = await client.post(
                    f"{ROUTER_URL}/v1/chat/completions",
                    json={
                        "model": current_model,
                        "messages": messages,
                        "max_tokens": 500,
                        "temperature": 0.7,
                    },
                    timeout=120.0,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    assistant_msg = data["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": assistant_msg})
                    print(f"\nAssistant: {assistant_msg}\n")
                else:
                    print(f"Error: HTTP {response.status_code}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test client for Multi-Model Server")
    parser.add_argument("mode", choices=["test", "chat"], default="test", nargs="?",
                       help="Run mode: 'test' for automated tests, 'chat' for interactive")
    args = parser.parse_args()
    
    if args.mode == "test":
        asyncio.run(run_tests())
    else:
        asyncio.run(interactive_chat())


if __name__ == "__main__":
    main()

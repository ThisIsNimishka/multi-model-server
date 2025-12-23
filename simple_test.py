"""
Simple test for the Windows server
"""
import requests
import json

BASE_URL = "http://localhost:8001"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_chat():
    """Test chat endpoint"""
    print("\nTesting chat endpoint...")
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "mistral",
                "messages": [
                    {"role": "user", "content": "Hello! What is 2+2? Answer briefly."}
                ],
                "max_tokens": 50,
                "temperature": 0.7,
            },
            timeout=120,
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  Response: {content}")
            return True
        else:
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def interactive_chat():
    """Interactive chat mode"""
    print("\n" + "="*50)
    print("Interactive Chat (type 'quit' to exit)")
    print("="*50 + "\n")
    
    messages = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.7,
                },
                timeout=120,
            )
            
            if response.status_code == 200:
                data = response.json()
                assistant_msg = data["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": assistant_msg})
                print(f"\nAssistant: {assistant_msg}\n")
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        interactive_chat()
    else:
        print("="*50)
        print("Multi-Model Server Test")
        print("="*50 + "\n")
        
        health_ok = test_health()
        if health_ok:
            chat_ok = test_chat()
            print("\n" + "="*50)
            print(f"Health: {'✓ OK' if health_ok else '✗ FAILED'}")
            print(f"Chat:   {'✓ OK' if chat_ok else '✗ FAILED'}")
            print("="*50)
        else:
            print("\nServer not running. Start it with:")
            print("  python windows_server.py --model mistral --port 8001")

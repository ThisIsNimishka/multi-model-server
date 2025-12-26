# ============================================
# Flash Attention Test Script
# ============================================
# Tests if Flash Attention is working and compares speed
# Usage: python test_flash_attention.py
# ============================================

import torch
import time
import sys

print("=" * 50)
print("  Flash Attention 2 - Test Script")
print("=" * 50)
print()

# ============================================
# 1. Check GPU
# ============================================
print("[1] GPU Information")
print("-" * 50)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print(f"    Flash Attn Compatible: {'Yes' if props.major >= 8 else 'No'}")
else:
    print("  No CUDA GPU available!")
    sys.exit(1)

print()

# ============================================
# 2. Check Flash Attention Package
# ============================================
print("[2] Flash Attention Package")
print("-" * 50)

flash_attn_available = False
try:
    import flash_attn
    print(f"  ✓ flash-attn installed: v{flash_attn.__version__}")
    flash_attn_available = True
except ImportError:
    print("  ✗ flash-attn not installed")

# Check PyTorch SDPA
sdpa_available = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
print(f"  {'✓' if sdpa_available else '✗'} PyTorch SDPA: {'Available' if sdpa_available else 'Not Available'}")

print()

# ============================================
# 3. Check Transformers Support
# ============================================
print("[3] Transformers Support")
print("-" * 50)

try:
    import transformers
    print(f"  Transformers version: {transformers.__version__}")
    
    # Check for Flash Attention support
    from transformers.utils import is_flash_attn_2_available
    print(f"  Flash Attention 2 in Transformers: {'✓ Available' if is_flash_attn_2_available() else '✗ Not Available'}")
except Exception as e:
    print(f"  Error checking transformers: {e}")

print()

# ============================================
# 4. Test SDPA Performance
# ============================================
print("[4] SDPA Performance Test")
print("-" * 50)

if sdpa_available:
    # Test parameters
    batch_size = 4
    seq_len = 512
    num_heads = 32
    head_dim = 128
    
    print(f"  Test config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}")
    
    # Create test tensors
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(3):
        _ = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.cuda.synchronize()
    
    # Benchmark SDPA
    start = time.perf_counter()
    for _ in range(10):
        output = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    torch.cuda.synchronize()
    sdpa_time = (time.perf_counter() - start) / 10
    
    print(f"  SDPA avg time: {sdpa_time*1000:.2f} ms")
    
    # Benchmark standard attention for comparison
    start = time.perf_counter()
    for _ in range(10):
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        output_std = torch.matmul(attn_weights, value)
    torch.cuda.synchronize()
    std_time = (time.perf_counter() - start) / 10
    
    print(f"  Standard attention avg time: {std_time*1000:.2f} ms")
    print(f"  Speedup: {std_time/sdpa_time:.2f}x faster with SDPA")
else:
    print("  SDPA not available, skipping test")

print()

# ============================================
# 5. Test Model Loading with Flash Attention
# ============================================
print("[5] Model Loading Test")
print("-" * 50)

test_model = input("Test loading a model with Flash Attention? (mistral/qwen-vl/skip): ").strip().lower()

if test_model in ["mistral", "qwen-vl"]:
    MODEL_PATHS = {
        "mistral": r"D:\AI_MODELS\Mistral-7B-Instruct-v0.3",
        "qwen-vl": r"D:\AI_MODELS\Qwen2.5-VL-7B-Instruct",
    }
    
    model_path = MODEL_PATHS[test_model]
    print(f"  Loading {test_model} from {model_path}...")
    
    # Determine attention implementation
    if flash_attn_available:
        attn_impl = "flash_attention_2"
    elif sdpa_available:
        attn_impl = "sdpa"
    else:
        attn_impl = "eager"
    
    print(f"  Using attention: {attn_impl}")
    
    try:
        start = time.perf_counter()
        
        if test_model == "mistral":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
        else:  # qwen-vl
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                attn_implementation=attn_impl,
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(model_path)
        
        load_time = time.perf_counter() - start
        print(f"  ✓ Model loaded in {load_time:.2f}s")
        
        # Check memory
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU Memory used: {memory_used:.2f} GB")
        
        # Quick generation test
        print("  Running quick generation test...")
        
        if test_model == "mistral":
            inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
            
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=50, use_cache=True)
            gen_time = time.perf_counter() - start
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens = outputs.shape[1]
            
        else:  # qwen-vl
            messages = [{"role": "user", "content": "Hello, how are you?"}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], return_tensors="pt").to(model.device)
            
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=50, use_cache=True)
            gen_time = time.perf_counter() - start
            
            response = processor.decode(outputs[0], skip_special_tokens=True)
            tokens = outputs.shape[1]
        
        print(f"  ✓ Generated {tokens} tokens in {gen_time:.2f}s ({tokens/gen_time:.1f} tok/s)")
        print(f"  Response preview: {response[:100]}...")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")

print()

# ============================================
# Summary
# ============================================
print("=" * 50)
print("  Summary")
print("=" * 50)
print()
print(f"  Flash Attention 2 (flash-attn): {'✓ Installed' if flash_attn_available else '✗ Not Installed'}")
print(f"  PyTorch SDPA:                   {'✓ Available' if sdpa_available else '✗ Not Available'}")
print()

if flash_attn_available:
    print("  Recommendation: Use attn_implementation='flash_attention_2'")
elif sdpa_available:
    print("  Recommendation: Use attn_implementation='sdpa' (built-in, no install needed)")
else:
    print("  Recommendation: Update PyTorch to 2.0+ for SDPA support")

print()
print("=" * 50)

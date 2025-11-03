# Execution Guide for DGX Spark NVFP4 Quantization

## Pre-Execution Checklist

On `spark-alpha.local`, verify:

- [ ] SSH access working: `ssh spark-alpha.local`
- [ ] GPU available: `nvidia-smi` shows GB10/Blackwell
- [ ] CUDA version: `nvcc --version` shows 12.0+
- [ ] Disk space: `df -h /` shows >150GB free
- [ ] Memory: `free -h` shows ~119GB+ available
- [ ] UV installed: `which uv` shows `/home/exobit/.local/bin/uv`
- [ ] Python 3.12: `python --version` or `python3 --version`

## Step-by-Step Execution

### Phase 0: Environment Setup (5 minutes)

```bash
# SSH to spark-alpha
ssh spark-alpha.local

# Navigate to project
cd /home/exobit/development/sgl/nvfp4-quantization

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv sync

# Verify installation
uv pip list | grep -E "torch|transformers|modelopt"
```

**Expected output:**
```
torch              2.5.0+
transformers       4.46.0+
nvidia-modelopt    0.18.0+
```

**Verification checkpoint:**
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] PyTorch with CUDA support available

### Phase 1: Download Model (10-30 minutes)

```bash
# Login to Hugging Face (if needed for gated models)
huggingface-cli login
# Paste your HF token when prompted

# Pre-download model to cache
huggingface-cli download qingy2024/Qwen3-VLTO-32B-Instruct

# Verify download
ls -lh ~/.cache/huggingface/hub/models--qingy2024--Qwen3-VLTO-32B-Instruct/
```

**Expected output:**
```
snapshots/
blobs/
refs/
```

**Verification checkpoint:**
- [ ] Model downloaded to HF cache (~64GB)
- [ ] All model shards present

### Phase 2: Calibration Data Preparation (10-15 minutes)

```bash
# Run calibration script
uv run scripts/01_prepare_calibration_data.py
```

**What to watch for:**
- Progress updates every 50 samples
- `✓ Sample processing complete!`
- `✓ Calibration data saved successfully`

**Expected output location:**
```
calibration-data/
├── calibration.json           # ~100-200 MB
└── calibration_metadata.json
```

**Verification checkpoint:**
- [ ] Script completed without errors
- [ ] `calibration.json` created
- [ ] 512 samples prepared
- [ ] Log file created in `logs/calibration_prep_*.log`

**Troubleshooting:**
```bash
# If dataset download is slow
# Check: tail -f logs/calibration_prep_*.log

# If OOM during data prep
# Reduce samples in configs/quantization_config.json to 256
```

### Phase 3: NVFP4 Quantization (60-90 minutes)

**⚠️ CRITICAL: This is the longest step. Do not interrupt.**

```bash
# Start quantization
uv run scripts/02_quantize_to_nvfp4.py

# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi
```

**Progress indicators:**
- `[1/6] Loading calibration data...` (~1 min)
- `[2/6] Loading model...` (~5-10 min)
- `[3/6] Preparing calibration function...` (~1 min)
- `[4/6] Quantizing model to NVFP4...` (~45-75 min)
  - Watch for: "Calibrated X/512 samples..." updates
- `[5/6] Exporting quantized model...` (~5 min)
- `[6/6] Saving quantization metadata...` (~1 min)

**Expected memory usage during quantization:**
- Peak: 60-80GB GPU memory
- After quantization: 20-25GB

**Expected output:**
```
quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/
├── config.json
├── hf_quant_config.json          # CRITICAL: Contains NVFP4 config
├── model-00001-of-*.safetensors  # ~18-20GB total
├── tokenizer.json
├── tokenizer_config.json
└── quantization_metadata.json
```

**Verification checkpoint:**
- [ ] Quantization completed without errors
- [ ] `hf_quant_config.json` exists
- [ ] Model files ~70% smaller than original
- [ ] Memory reduction logged (should show ~70% reduction)
- [ ] Log file: `logs/nvfp4_quantization_*.log`

**Troubleshooting:**
```bash
# If OOM during quantization
# Option 1: Reduce calibration samples
nano configs/quantization_config.json
# Set "calibration_samples": 256

# Option 2: Check GPU memory
nvidia-smi
# Kill any competing processes

# If quantization hangs
# Check log file for last operation
tail -f logs/nvfp4_quantization_*.log
```

### Phase 4: Inference Testing (5-10 minutes)

```bash
# Test quantized model
uv run scripts/03_test_inference.py
```

**What to watch for:**
- 5 test cases running
- Each should complete in 5-15 seconds
- Responses should be coherent
- No CUDA errors

**Expected output:**
```
Test Results Summary:
  Total tests: 5
  Successful: 5
  Failed: 0

Performance Metrics (Average):
  Time per test: X.XXs
  Tokens per test: ~XXX
  Speed: XX.XX tokens/s

Memory Usage:
  Allocated: ~20 GB
  Reserved: ~25 GB
```

**Verification checkpoint:**
- [ ] All 5 tests passed
- [ ] Responses are coherent and relevant
- [ ] Speed >10 tokens/second
- [ ] Memory <25GB
- [ ] Results saved: `logs/inference_results_*.json`

### Phase 5: Performance Benchmark (10-15 minutes)

```bash
# Run comprehensive benchmark
uv run scripts/04_benchmark.py
```

**Test sequence:**
1. Model loading and warmup (3 iterations)
2. Throughput test (10 iterations)
3. Latency test (10 iterations)
4. Sequence length scaling (4 length tests)
5. Memory efficiency analysis

**Expected results:**
```
Performance Summary:
  Throughput: 15-30 tokens/s
  Latency (avg): 5-15s
  Latency (P95): <20s
  Memory usage: 18-22 GB
  Memory efficiency: 15-20% of GPU
```

**Verification checkpoint:**
- [ ] All benchmarks completed
- [ ] Throughput ≥15 tokens/s
- [ ] Memory <25GB
- [ ] Results saved: `logs/benchmark_results_*.json`

## Post-Execution Validation

### Final Verification

```bash
# 1. Check quantized model structure
ls -lh quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/

# 2. Verify NVFP4 configuration
cat quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/hf_quant_config.json | grep -i "nvfp4\|quant"

# 3. Check total model size
du -sh quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/

# Expected: ~20-25GB (vs ~64GB original)

# 4. Review all logs
ls -lht logs/

# 5. Check metadata
cat quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/quantization_metadata.json
```

### Quality Checklist

- [ ] **Model files exist**: `hf_quant_config.json` and `*.safetensors` present
- [ ] **Size reduction**: Quantized model ~70% smaller than original
- [ ] **NVFP4 format**: `hf_quant_config.json` contains NVFP4 configuration
- [ ] **Inference works**: All test cases passed
- [ ] **Performance acceptable**: >15 tokens/s throughput
- [ ] **Memory efficient**: <25GB GPU memory
- [ ] **Quality maintained**: Responses are coherent and accurate
- [ ] **Logs complete**: All operations logged with no errors

## Deployment

### Option 1: Direct Python Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/home/exobit/development/sgl/nvfp4-quantization/quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "/home/exobit/development/sgl/nvfp4-quantization/quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4",
    trust_remote_code=True
)

# Generate
prompt = "Explain how transformers work in machine learning:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option 2: vLLM Server Deployment

```bash
# Pull vLLM container
docker pull lmsysorg/sglang:spark

# Start server
docker run -d --gpus all \
  --name qwen3-nvfp4-server \
  --shm-size 32g \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /home/exobit/development/sgl/nvfp4-quantization/quantized-output:/workspace \
  --ipc=host \
  lmsysorg/sglang:spark \
  python -m vllm.entrypoints.openai.api_server \
    --model /workspace/Qwen3-VLTO-32B-Instruct-NVFP4 \
    --trust-remote-code \
    --quantization fp4 \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192

# Test API
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/workspace/Qwen3-VLTO-32B-Instruct-NVFP4",
    "prompt": "Explain quantum computing:",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### Option 3: Multi-Node (Both Sparks)

If you want to use both `spark-alpha.local` and `spark-omega.local`:

```bash
# On spark-alpha.local, setup networking
wget https://github.com/NVIDIA/dgx-spark-playbooks/raw/main/nvidia/connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks
./discover-sparks

# Follow prompts to configure spark-omega.local connection

# Deploy with tensor parallelism
docker run -d --gpus all \
  --name qwen3-nvfp4-cluster \
  --shm-size 64g \
  --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /home/exobit/development/sgl/nvfp4-quantization/quantized-output:/workspace \
  lmsysorg/sglang:spark \
  python -m sglang.launch_server \
    --model /workspace/Qwen3-VLTO-32B-Instruct-NVFP4 \
    --tp-size 2 \
    --trust-remote-code \
    --port 8000
```

## Timeline Summary

| Phase | Duration | Can Skip? |
|-------|----------|-----------|
| Environment setup | 5 min | No |
| Model download | 10-30 min | Only if cached |
| Calibration prep | 10-15 min | No |
| **Quantization** | **60-90 min** | **No** |
| Inference testing | 5-10 min | No |
| Benchmarking | 10-15 min | Optional |
| **Total** | **2-3 hours** | |

## Troubleshooting Reference

### Common Issues

**Issue: `ImportError: cannot import name 'mtq'`**
```bash
# Reinstall nvidia-modelopt
uv pip install --force-reinstall nvidia-modelopt
```

**Issue: CUDA out of memory**
```bash
# Check what's using GPU
nvidia-smi

# Free memory
sudo fuser -v /dev/nvidia*
kill <PID>

# Reduce calibration samples
nano configs/quantization_config.json
# Set "calibration_samples": 256
```

**Issue: Model download fails**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--qingy2024--Qwen3-VLTO-32B-Instruct/
huggingface-cli download qingy2024/Qwen3-VLTO-32B-Instruct --resume-download
```

**Issue: Quantization very slow**
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Should show ~90-100% GPU utilization
# If low, check for:
# - Thermal throttling: nvidia-smi -q -d TEMPERATURE
# - Power limits: nvidia-smi -q -d POWER
```

## Success Criteria

✅ **Quantization is successful if ALL of these are true:**

1. ✓ `quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/` directory exists
2. ✓ `hf_quant_config.json` contains NVFP4 configuration
3. ✓ Model size is 18-25GB (check with `du -sh`)
4. ✓ All 5 inference tests pass
5. ✓ Benchmark throughput >15 tokens/s
6. ✓ Memory usage <25GB
7. ✓ No errors in any log files
8. ✓ Model generates coherent, relevant responses

## Next Steps After Success

1. **Production deployment**: Choose deployment option (Python, vLLM, or multi-node)
2. **Quality evaluation**: Run on your specific use cases
3. **Performance tuning**: Adjust generation parameters for your workload
4. **Monitoring**: Set up logging and metrics collection
5. **Backup**: Archive the quantized model

## Support Files

All execution logs and results are in:
```
logs/
├── calibration_prep_YYYYMMDD_HHMMSS.log
├── nvfp4_quantization_YYYYMMDD_HHMMSS.log
├── inference_test_YYYYMMDD_HHMMSS.log
├── benchmark_YYYYMMDD_HHMMSS.log
├── inference_results_YYYYMMDD_HHMMSS.json
└── benchmark_results_YYYYMMDD_HHMMSS.json
```

Refer to these files for detailed diagnostics if anything fails.

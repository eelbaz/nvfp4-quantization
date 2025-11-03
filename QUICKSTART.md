# Quick Start Guide - NVFP4 Quantization

## One-Command Setup

```bash
cd nvfp4-quantization && uv venv && source .venv/bin/activate && uv sync
```

## Run Complete Pipeline

### Step 1: Prepare Calibration (10-15 min)
```bash
uv run scripts/01_prepare_calibration_data.py
```
**What it does:** Downloads C4 dataset and prepares 512 calibration samples

### Step 2: Quantize Model (60-90 min)
```bash
uv run scripts/02_quantize_to_nvfp4.py
```
**What it does:** Loads 32B model, calibrates, and quantizes to NVFP4 (~70% memory reduction)

### Step 3: Test Inference (5-10 min)
```bash
uv run scripts/03_test_inference.py
```
**What it does:** Runs 5 test prompts to verify model quality

### Step 4: Benchmark (10-15 min)
```bash
uv run scripts/04_benchmark.py
```
**What it does:** Measures throughput, latency, and memory efficiency

## Expected Results

After successful completion:

| Metric | Value |
|--------|-------|
| Memory | ~18-22 GB (vs 64GB BF16) |
| Speed | 15-30 tokens/s |
| TTFT | 1-3 seconds |
| Accuracy | <1% loss |

## Files Created

```
quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/
├── config.json                    # Model configuration
├── hf_quant_config.json          # NVFP4 quantization config
├── model-00001-of-*.safetensors  # Quantized weights
├── tokenizer.json                # Tokenizer
└── quantization_metadata.json    # Quantization details
```

## Quick Verification

```bash
# Check quantized model size
du -sh quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/

# View quantization config
cat quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/hf_quant_config.json

# Check latest logs
ls -lt logs/ | head -5
```

## Troubleshooting

**OOM during quantization?**
```bash
# Reduce calibration samples in configs/quantization_config.json
# Change "calibration_samples": 512 to "calibration_samples": 256
```

**Model not downloading?**
```bash
# Pre-download to HF cache
huggingface-cli download qingy2024/Qwen3-VLTO-32B-Instruct
```

**Check GPU status:**
```bash
nvidia-smi
```

## Next Steps

After quantization, deploy with:

**Option 1: Python API**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4",
    device_map="auto",
    trust_remote_code=True
)
```

**Option 2: vLLM Server**
```bash
docker run --gpus all -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/quantized-output:/workspace \
  lmsysorg/sglang:spark \
  python -m vllm.entrypoints.openai.api_server \
    --model /workspace/Qwen3-VLTO-32B-Instruct-NVFP4 \
    --quantization fp4 --port 8000
```

## Timeline

Total time: **~2-3 hours**
- Setup: 5 minutes
- Calibration prep: 15 minutes
- Quantization: 60-90 minutes
- Testing: 15 minutes
- Benchmarking: 15 minutes

## Support

Check logs in `logs/` directory for detailed output and error messages.

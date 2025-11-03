# NVFP4 Quantization for Qwen3-VLTO-32B-Instruct

Production-ready NVFP4 quantization workflow for NVIDIA DGX Spark (GB10 Blackwell) using Docker containers.

## Overview

This project quantizes the [qingy2024/Qwen3-VLTO-32B-Instruct](https://huggingface.co/qingy2024/Qwen3-VLTO-32B-Instruct) model to NVFP4 (4-bit floating point) format for optimized inference on DGX Spark systems with Blackwell GB10 GPUs.

### NVFP4 Format

NVFP4 is NVIDIA TensorRT Model Optimizer's 4-bit floating point quantization optimized for Blackwell GPUs:
- Two-level scaling: E4M3 FP8 scaling per 16-value block plus global FP32 tensor scale
- Hardware acceleration via Tensor Cores on GB10
- Memory efficiency: approximately 70 percent reduction vs BF16 (64GB to 18-20GB)
- Minimal accuracy degradation: less than 1 percent vs original model
- Tested inference speed: approximately 10 tokens per second on DGX Spark

### System Requirements

**Hardware:**
- NVIDIA DGX Spark with GB10 (Grace Blackwell Superchip)
- 128GB unified memory
- aarch64 (ARM64) architecture

**Software:**
- Docker with NVIDIA Container Toolkit
- CUDA 12.0 or later
- Approximately 150GB free disk space

## Quick Start

### 1. Prepare Calibration Data

```bash
./docker-run.sh scripts/01_prepare_calibration_data.py
```

This downloads the C4 dataset and prepares 512 calibration samples (approximately 10-15 minutes).

### 2. Quantize Model to NVFP4

```bash
./docker-run.sh scripts/02_quantize_to_nvfp4.py
```

This performs the full quantization workflow (approximately 60-90 minutes):
- Loads model in BF16
- Runs calibration forward pass
- Applies NVFP4 quantization
- Exports to `quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/`

### 3. Test Inference

```bash
./docker-run-vllm.sh scripts/03_test_inference_vllm.py
```

Validates the quantized model with 5 test cases (approximately 5-10 minutes).

### 4. Production Deployment with Open WebUI

Start a vLLM server with OpenAI-compatible API:

```bash
./start-vllm-server.sh
```

The server will:
- Run as Docker container `vllm-nvfp4-server`
- Expose API at `http://localhost:8355/v1`
- Integrate with Open WebUI automatically
- Provide approximately 10 tokens per second throughput

**Management Commands:**
```bash
# View logs
docker logs -f vllm-nvfp4-server

# Stop server
./stop-vllm-server.sh

# Restart server
docker restart vllm-nvfp4-server
```

**Test API:**
```bash
# List models
curl http://localhost:8355/v1/models

# Test completion
curl http://localhost:8355/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/quantized", "prompt": "Hello", "max_tokens": 50}'
```

## Docker-Based Workflow

This project uses NVIDIA official Docker containers for compatibility with DGX Spark aarch64 architecture:

**For Quantization:**
- Container: `nvcr.io/nvidia/pytorch:25.10-py3`
- Includes: PyTorch 2.9.0, CUDA 13.0, nvidia-modelopt

**For Inference:**
- Container: `nvcr.io/nvidia/vllm:25.10-py3`
- Includes: vLLM 0.10.2, modelopt quantization support

Both wrapper scripts (`docker-run.sh` and `docker-run-vllm.sh`) automatically:
- Mount the project directory
- Mount HuggingFace cache
- Install required dependencies
- Set up proper CUDA paths

## Project Structure

```
nvfp4-quantization/
├── README.md                           # This file
├── docker-run.sh                       # Quantization container wrapper
├── docker-run-vllm.sh                  # Inference container wrapper
├── configs/
│   └── quantization_config.json        # Quantization parameters
├── scripts/
│   ├── 01_prepare_calibration_data.py  # Calibration data prep
│   ├── 02_quantize_to_nvfp4.py         # Main quantization script
│   ├── 03_test_inference_vllm.py       # vLLM inference testing
│   └── 04_benchmark.py                 # Performance benchmarks
├── calibration-data/                   # Generated calibration samples
├── quantized-output/                   # Quantized model output
│   └── Qwen3-VLTO-32B-Instruct-NVFP4/
└── logs/                               # Execution logs and results
```

## Configuration

Edit `configs/quantization_config.json` to customize parameters:

```json
{
  "model_name": "qingy2024/Qwen3-VLTO-32B-Instruct",
  "quantization_format": "NVFP4",
  "calibration_samples": 512,
  "calibration_max_length": 2048,
  "hf_cache_dir": "~/.cache/huggingface",
  "output_dir": "./quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4"
}
```

**Key Parameters:**
- `calibration_samples`: Number of calibration samples (default: 512)
- `calibration_max_length`: Maximum tokens per sample (default: 2048)
- `hf_cache_dir`: HuggingFace cache directory (default: `~/.cache/huggingface`)

## Inference with vLLM

The quantized model must be loaded with vLLM using modelopt quantization support:

```python
from vllm import LLM, SamplingParams

# Load NVFP4 quantized model
llm = LLM(
    model="./quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4",
    quantization="modelopt",  # Required for NVFP4
    trust_remote_code=True,
    gpu_memory_utilization=0.9
)

# Generate
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(["Explain quantum computing:"], sampling_params)
print(outputs[0].outputs[0].text)
```

**Important:** Standard `AutoModelForCausalLM.from_pretrained()` will not work with NVFP4 models. You must use vLLM 0.6.5 or later with `quantization="modelopt"`.

## Expected Performance

On DGX Spark GB10:

| Metric | Value |
|--------|-------|
| Memory Usage | 18-22 GB |
| Inference Speed | 10 tokens/s |
| Model Loading | approximately 2 minutes |
| Accuracy vs BF16 | less than 1 percent degradation |
| Memory Reduction | approximately 70 percent |

## Verification Checklist

After quantization, verify:

- Quantized model directory exists at `quantized-output/Qwen3-VLTO-32B-Instruct-NVFP4/`
- `hf_quant_config.json` contains `"quant_algo": "NVFP4"`
- Five safetensors shards present (model-00001 through model-00005)
- Inference test passes all 5 test cases
- Model generates coherent responses

## Troubleshooting

### Out of Memory During Quantization

Reduce calibration samples in config:
```json
"calibration_samples": 256
```

### vLLM Import Errors

Ensure you use `docker-run-vllm.sh` for inference scripts, not `docker-run.sh`. The vLLM container includes the correct dependencies.

### Model Not Found

Verify the model downloaded to HuggingFace cache:
```bash
ls ~/.cache/huggingface/hub/models--qingy2024--Qwen3-VLTO-32B-Instruct/
```

## Environment Variables

Both Docker wrapper scripts support these environment variables:

- `HF_CACHE_DIR`: Override HuggingFace cache location (default: `$HOME/.cache/huggingface`)

Example:
```bash
HF_CACHE_DIR=/data/hf-cache ./docker-run.sh scripts/01_prepare_calibration_data.py
```

## Logs and Results

All scripts generate detailed logs in `logs/`:

- `calibration_prep_*.log` - Calibration data preparation
- `nvfp4_quantization_*.log` - Quantization process
- `inference_test_vllm_*.log` - Inference testing
- `inference_results_vllm_*.json` - Test results (JSON format)

## References

- [NVIDIA DGX Spark Documentation](https://docs.nvidia.com/dgx-spark/)
- [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3-VL Model](https://huggingface.co/qingy2024/Qwen3-VLTO-32B-Instruct)

## License

This quantization workflow is provided as-is. The underlying model follows its own license terms.

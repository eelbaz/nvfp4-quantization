---
language:
- en
license: apache-2.0
tags:
- text-generation
- quantization
- nvfp4
- nvidia
- dgx-spark
- blackwell
- model_hub_mixin
- pytorch_model_hub_mixin
base_model: qingy2024/Qwen3-VLTO-32B-Instruct
inference: false
---

# Qwen3-VLTO-32B-Instruct-NVFP4

This is an NVFP4 quantized version of [qingy2024/Qwen3-VLTO-32B-Instruct](https://huggingface.co/qingy2024/Qwen3-VLTO-32B-Instruct), optimized for NVIDIA DGX Spark systems with Blackwell GB10 GPUs.

## Model Description

- **Base Model:** qingy2024/Qwen3-VLTO-32B-Instruct
- **Quantization Format:** NVFP4 (4-bit floating point)
- **Target Hardware:** NVIDIA DGX Spark (Grace Blackwell Superchip)
- **Quantization Tool:** NVIDIA TensorRT Model Optimizer v0.35.1
- **Model Size:** Approximately 20 GB (68% reduction from BF16)

## Performance Characteristics

### Memory Efficiency

| Model Version | Memory Usage | Reduction |
|--------------|--------------|-----------|
| BF16 (Original) | 61.03 GB | Baseline |
| NVFP4 (This model) | 19.42 GB | 68.2% |

### Inference Speed

| Model Version | Throughput | Relative Performance |
|--------------|------------|---------------------|
| BF16 (Original) | 3.65 tokens/s | Baseline |
| NVFP4 (This model) | 9.99 tokens/s | 2.74x faster |

**Test Configuration:**
- Hardware: NVIDIA DGX Spark GB10
- Framework: vLLM 0.10.2
- Max Model Length: 8192 tokens
- GPU Memory Utilization: 90%

## Quantization Details

### NVFP4 Format

NVFP4 is NVIDIA's 4-bit floating point quantization format featuring:
- **Two-level scaling:** E4M3 FP8 scaling per 16-value block + global FP32 tensor scale
- **Hardware acceleration:** Optimized for Tensor Cores on Blackwell GB10 GPUs
- **Group size:** 16
- **Minimal accuracy degradation:** Less than 1% vs original model
- **Excluded modules:** lm_head (kept in higher precision)

### Calibration

- **Dataset:** C4 (Colossal Clean Crawled Corpus)
- **Calibration samples:** 512
- **Maximum sequence length:** 2048 tokens
- **Method:** Post-training quantization with activation calibration

## Usage

### Requirements

- NVIDIA DGX Spark or compatible Blackwell GPU
- vLLM >= 0.6.5
- nvidia-modelopt[hf]

### Loading the Model

**IMPORTANT:** This model must be loaded with vLLM using the `modelopt` quantization parameter. Standard HuggingFace `AutoModelForCausalLM` will not work.

```python
from vllm import LLM, SamplingParams

# Load NVFP4 quantized model
llm = LLM(
    model="Ex0bit/Qwen3-VLTO-32B-Instruct-NVFP4",
    quantization="modelopt",  # Required for NVFP4
    trust_remote_code=True,
    gpu_memory_utilization=0.9
)

# Generate
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
outputs = llm.generate(["Explain quantum computing in simple terms:"], sampling_params)
print(outputs[0].outputs[0].text)
```

### Environment Variables

You can optionally set:
- `HF_CACHE_DIR`: Override HuggingFace cache location

## Limitations

- **Hardware specific:** Optimized for NVIDIA Blackwell architecture (GB10)
- **vLLM required:** Cannot be loaded with standard transformers library
- **Quantization artifacts:** Minor precision loss (<1%) compared to BF16 original

## Intended Use

This model is intended for:
- High-throughput inference on NVIDIA DGX Spark systems
- Production deployments requiring memory-efficient models
- Research on quantization techniques for large language models

## Training and Quantization

### Base Model Training

See the [original model card](https://huggingface.co/qingy2024/Qwen3-VLTO-32B-Instruct) for base model training details.

### Quantization Process

1. **Model Loading:** Original model loaded in BF16 precision
2. **Calibration:** 512 samples from C4 dataset for activation statistics
3. **Quantization:** NVFP4 format applied using NVIDIA modelopt
4. **Export:** Saved in HuggingFace safetensors format

**Quantization Time:** Approximately 60-90 minutes on DGX Spark

## Evaluation

### Test Results

All 5 inference tests passed successfully:
- Technical explanation generation
- Code generation
- Mathematical reasoning
- Creative writing
- Instruction following

**Average performance:** 9.99 tokens/s on DGX Spark GB10

## Citation

If you use this quantized model, please cite:

```bibtex
@misc{qwen3vlto32b-nvfp4,
  author = {Ex0bit},
  title = {Qwen3-VLTO-32B-Instruct-NVFP4: NVFP4 Quantized Model for DGX Spark},
  year = {2025},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/Ex0bit/Qwen3-VLTO-32B-Instruct-NVFP4}},
}
```

And the original base model:

```bibtex
@misc{qingy2024qwen3vlto,
  author = {qingy2024},
  title = {Qwen3-VLTO-32B-Instruct},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\url{https://huggingface.co/qingy2024/Qwen3-VLTO-32B-Instruct}},
}
```

## References

- [NVIDIA TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [vLLM Documentation](https://docs.vllm.ai/)
- [NVIDIA DGX Spark Documentation](https://docs.nvidia.com/dgx-spark/)
- [Quantization GitHub Repository](https://github.com/Ex0bit/nvfp4-quantization)

## License

This quantized model inherits the license from the base model. Please refer to the [original model's license](https://huggingface.co/qingy2024/Qwen3-VLTO-32B-Instruct) for details.

## Model Card Authors

- Ex0bit (@Ex0bit)

## Acknowledgments

- NVIDIA for TensorRT Model Optimizer and DGX Spark hardware
- qingy2024 for the base Qwen3-VLTO-32B-Instruct model
- The vLLM team for high-performance inference framework

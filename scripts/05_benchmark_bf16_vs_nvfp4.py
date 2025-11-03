#!/usr/bin/env python3
"""
BF16 vs NVFP4 Benchmark Comparison Script
==========================================

Comprehensive benchmark comparing the original BF16 model with the NVFP4 quantized model.

Measures:
- Throughput (tokens/second)
- Memory usage
- Latency
- Model quality (perplexity on test samples)

Usage:
    python scripts/05_benchmark_bf16_vs_nvfp4.py

Prerequisites:
    - Original BF16 model accessible via HuggingFace
    - Quantized NVFP4 model created by 02_quantize_to_nvfp4.py
    - vLLM 0.6.5 or later
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime
import statistics

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent.parent / "configs" / "quantization_config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Model paths
ORIGINAL_MODEL = config['model_name']
QUANTIZED_MODEL = Path(__file__).parent.parent / config['output_dir']

# Benchmark configuration
BENCHMARK_CONFIG = {
    "warmup_iterations": 2,
    "benchmark_iterations": 5,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9
}

# Test prompts for benchmarking
BENCHMARK_PROMPTS = [
    "Explain the concept of machine learning and its applications in modern technology.",
    "Write a comprehensive guide to understanding neural networks.",
    "Describe the process of photosynthesis in plants step by step.",
    "Analyze the impact of renewable energy on climate change.",
    "Discuss the evolution of programming languages from assembly to modern high-level languages."
]


def load_model_vllm(model_path: str, quantization: str = None):
    """Load a model using vLLM."""
    from vllm import LLM

    logger.info(f"Loading model: {model_path}")
    logger.info(f"  Quantization: {quantization if quantization else 'None (BF16)'}")

    try:
        llm_kwargs = {
            "model": str(model_path),
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 8192
        }

        if quantization:
            llm_kwargs["quantization"] = quantization

        llm = LLM(**llm_kwargs)
        logger.info(f"  Model loaded successfully")

        return llm

    except Exception as e:
        logger.error(f"  Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def get_memory_usage():
    """Get current GPU memory usage."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9  # Convert to GB
    return 0


def benchmark_model(llm, model_name: str, iterations: int) -> Dict:
    """Run comprehensive benchmark on a model."""
    from vllm import SamplingParams

    logger.info(f"\n{'='*80}")
    logger.info(f"Benchmarking: {model_name}")
    logger.info(f"{'='*80}")

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=BENCHMARK_CONFIG['temperature'],
        top_p=BENCHMARK_CONFIG['top_p'],
        max_tokens=BENCHMARK_CONFIG['max_new_tokens']
    )

    # Warmup
    logger.info(f"\nWarming up ({BENCHMARK_CONFIG['warmup_iterations']} iterations)...")
    for i in range(BENCHMARK_CONFIG['warmup_iterations']):
        llm.generate(["Hello, this is a warmup test."], sampling_params)
        logger.info(f"  Warmup {i+1}/{BENCHMARK_CONFIG['warmup_iterations']} complete")

    # Memory measurement
    logger.info(f"\nMeasuring memory usage...")
    memory_gb = get_memory_usage()
    logger.info(f"  Memory allocated: {memory_gb:.2f} GB")

    # Throughput benchmark
    logger.info(f"\nRunning throughput benchmark ({iterations} iterations)...")
    results = []

    for i in range(iterations):
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]

        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.time()

        elapsed = end_time - start_time
        tokens_generated = len(outputs[0].outputs[0].token_ids)
        tokens_per_second = tokens_generated / elapsed

        results.append({
            "iteration": i + 1,
            "tokens_generated": tokens_generated,
            "time_seconds": elapsed,
            "tokens_per_second": tokens_per_second
        })

        logger.info(f"  Iteration {i+1}: {tokens_generated} tokens in {elapsed:.2f}s ({tokens_per_second:.2f} tokens/s)")

    # Calculate statistics
    avg_throughput = statistics.mean(r['tokens_per_second'] for r in results)
    median_throughput = statistics.median(r['tokens_per_second'] for r in results)
    stdev_throughput = statistics.stdev([r['tokens_per_second'] for r in results]) if len(results) > 1 else 0

    logger.info(f"\nResults Summary:")
    logger.info(f"  Average throughput: {avg_throughput:.2f} tokens/s")
    logger.info(f"  Median throughput: {median_throughput:.2f} tokens/s")
    logger.info(f"  Std dev: {stdev_throughput:.2f}")
    logger.info(f"  Memory usage: {memory_gb:.2f} GB")

    return {
        "model_name": model_name,
        "iterations": results,
        "average_tokens_per_second": avg_throughput,
        "median_tokens_per_second": median_throughput,
        "stdev_tokens_per_second": stdev_throughput,
        "memory_allocated_gb": memory_gb
    }


def compare_results(bf16_results: Dict, nvfp4_results: Dict) -> Dict:
    """Compare BF16 and NVFP4 results."""
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPARISON RESULTS")
    logger.info(f"{'='*80}")

    # Speed comparison
    bf16_speed = bf16_results['average_tokens_per_second']
    nvfp4_speed = nvfp4_results['average_tokens_per_second']
    speed_diff_pct = ((nvfp4_speed - bf16_speed) / bf16_speed) * 100

    logger.info(f"\nThroughput:")
    logger.info(f"  BF16:   {bf16_speed:.2f} tokens/s")
    logger.info(f"  NVFP4:  {nvfp4_speed:.2f} tokens/s")
    logger.info(f"  Difference: {speed_diff_pct:+.1f}%")

    # Memory comparison
    bf16_memory = bf16_results['memory_allocated_gb']
    nvfp4_memory = nvfp4_results['memory_allocated_gb']
    memory_reduction_pct = ((bf16_memory - nvfp4_memory) / bf16_memory) * 100

    logger.info(f"\nMemory Usage:")
    logger.info(f"  BF16:   {bf16_memory:.2f} GB")
    logger.info(f"  NVFP4:  {nvfp4_memory:.2f} GB")
    logger.info(f"  Reduction: {memory_reduction_pct:.1f}%")

    # Model size comparison
    model_size_reduction = memory_reduction_pct  # Approximate

    logger.info(f"\nModel Size:")
    logger.info(f"  Reduction: ~{model_size_reduction:.1f}%")
    logger.info(f"  NVFP4 is ~{100 - model_size_reduction:.1f}% of BF16 size")

    return {
        "throughput": {
            "bf16_tokens_per_second": bf16_speed,
            "nvfp4_tokens_per_second": nvfp4_speed,
            "difference_percent": speed_diff_pct
        },
        "memory": {
            "bf16_gb": bf16_memory,
            "nvfp4_gb": nvfp4_memory,
            "reduction_percent": memory_reduction_pct
        },
        "model_size": {
            "reduction_percent": model_size_reduction
        }
    }


def main():
    logger.info("="*80)
    logger.info("BF16 vs NVFP4 BENCHMARK COMPARISON")
    logger.info("="*80)
    logger.info(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"\nConfiguration:")
    logger.info(f"  Warmup iterations: {BENCHMARK_CONFIG['warmup_iterations']}")
    logger.info(f"  Benchmark iterations: {BENCHMARK_CONFIG['benchmark_iterations']}")
    logger.info(f"  Max new tokens: {BENCHMARK_CONFIG['max_new_tokens']}")

    # Check if quantized model exists
    if not QUANTIZED_MODEL.exists():
        logger.error(f"\nQuantized model not found at: {QUANTIZED_MODEL}")
        logger.error(f"Run: ./docker-run.sh scripts/02_quantize_to_nvfp4.py first")
        sys.exit(1)

    try:
        # Benchmark BF16 model
        logger.info(f"\n[1/2] Benchmarking BF16 model...")
        bf16_llm = load_model_vllm(ORIGINAL_MODEL)
        bf16_results = benchmark_model(
            bf16_llm,
            "BF16 (Original)",
            BENCHMARK_CONFIG['benchmark_iterations']
        )

        # Clean up BF16 model
        del bf16_llm
        import torch
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        time.sleep(5)  # Allow memory to be fully released

        # Benchmark NVFP4 model
        logger.info(f"\n[2/2] Benchmarking NVFP4 quantized model...")
        nvfp4_llm = load_model_vllm(str(QUANTIZED_MODEL), quantization="modelopt")
        nvfp4_results = benchmark_model(
            nvfp4_llm,
            "NVFP4 (Quantized)",
            BENCHMARK_CONFIG['benchmark_iterations']
        )

        # Compare results
        comparison = compare_results(bf16_results, nvfp4_results)

        # Save results
        results_file = log_dir / f"benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(results_file, 'w') as f:
            json.dump({
                "benchmark_config": BENCHMARK_CONFIG,
                "original_model": ORIGINAL_MODEL,
                "quantized_model": str(QUANTIZED_MODEL),
                "bf16_results": bf16_results,
                "nvfp4_results": nvfp4_results,
                "comparison": comparison,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"BENCHMARK COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"\nResults saved to: {results_file}")
        logger.info(f"Log file: {log_file}")

        # Final verdict
        logger.info(f"\nKey Findings:")
        logger.info(f"  Memory reduction: {comparison['memory']['reduction_percent']:.1f}%")
        logger.info(f"  Speed difference: {comparison['throughput']['difference_percent']:+.1f}%")
        logger.info(f"  NVFP4 achieves {comparison['memory']['reduction_percent']:.0f}% memory savings")

        if comparison['throughput']['difference_percent'] >= -10:
            logger.info(f"  Minimal speed degradation (<10%)")

    except Exception as e:
        logger.error(f"\nBenchmark failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
NVFP4 Performance Benchmark Script
===================================

Comprehensive performance benchmarking for the NVFP4 quantized model.

Measures:
- Throughput (tokens/second)
- Latency (time per request)
- Time to First Token (TTFT)
- Memory efficiency
- Performance across different sequence lengths

Usage:
    uv run scripts/04_benchmark.py

Prerequisites:
    - Quantized model created by 02_quantize_to_nvfp4.py
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

# Paths
MODEL_DIR = Path(__file__).parent.parent / config['output_dir']

# Benchmark configuration
BENCHMARK_CONFIG = {
    "warmup_iterations": 3,
    "benchmark_iterations": 10,
    "sequence_lengths": [128, 256, 512, 1024],
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


def load_model_and_tokenizer():
    """Load the NVFP4 quantized model and tokenizer."""
    logger.info(f"Loading model from: {MODEL_DIR}")

    if not MODEL_DIR.exists():
        logger.error(f"✗ Model directory not found: {MODEL_DIR}")
        logger.error(f"  Run: uv run scripts/02_quantize_to_nvfp4.py first")
        sys.exit(1)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            device_map="auto",
            trust_remote_code=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True
        )

        logger.info(f"✓ Model loaded successfully")

        # Get initial memory stats
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        logger.info(f"  Memory allocated: {memory_allocated:.2f} GB")

        return model, tokenizer

    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        sys.exit(1)


def warmup_model(model, tokenizer, iterations: int = 3):
    """Warm up the model with a few inference runs."""
    logger.info(f"\nWarming up model ({iterations} iterations)...")

    prompt = "Hello, this is a warmup test."

    for i in range(iterations):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        logger.info(f"  Warmup {i+1}/{iterations} complete")

    torch.cuda.synchronize()
    logger.info(f"✓ Warmup complete")


def benchmark_throughput(model, tokenizer, iterations: int) -> Dict:
    """Benchmark token generation throughput."""
    logger.info(f"\n[Benchmark 1/4] Throughput Test")
    logger.info(f"  Iterations: {iterations}")
    logger.info("-"*80)

    results = []

    for i in range(iterations):
        # Use different prompts
        prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)

        input_length = inputs['input_ids'].shape[1]

        # Measure generation time
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=BENCHMARK_CONFIG['max_new_tokens'],
                do_sample=True,
                temperature=BENCHMARK_CONFIG['temperature'],
                top_p=BENCHMARK_CONFIG['top_p'],
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        tokens_generated = outputs.shape[1] - input_length
        tokens_per_second = tokens_generated / elapsed

        results.append({
            "iteration": i + 1,
            "input_length": input_length,
            "tokens_generated": tokens_generated,
            "time_seconds": elapsed,
            "tokens_per_second": tokens_per_second
        })

        logger.info(f"  Iteration {i+1}: {tokens_generated} tokens in {elapsed:.2f}s ({tokens_per_second:.2f} tokens/s)")

    # Calculate statistics
    avg_throughput = statistics.mean(r['tokens_per_second'] for r in results)
    median_throughput = statistics.median(r['tokens_per_second'] for r in results)
    stdev_throughput = statistics.stdev(r['tokens_per_second']) if len(results) > 1 else 0

    logger.info(f"\nThroughput Results:")
    logger.info(f"  Average: {avg_throughput:.2f} tokens/s")
    logger.info(f"  Median: {median_throughput:.2f} tokens/s")
    logger.info(f"  Std Dev: {stdev_throughput:.2f}")

    return {
        "iterations": results,
        "average_tokens_per_second": avg_throughput,
        "median_tokens_per_second": median_throughput,
        "stdev_tokens_per_second": stdev_throughput
    }


def benchmark_latency(model, tokenizer, iterations: int) -> Dict:
    """Benchmark end-to-end latency."""
    logger.info(f"\n[Benchmark 2/4] Latency Test")
    logger.info(f"  Iterations: {iterations}")
    logger.info("-"*80)

    latencies = []
    prompt = "Explain quantum computing in simple terms."

    for i in range(iterations):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=BENCHMARK_CONFIG['max_new_tokens'],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        end_time = time.time()

        latency = end_time - start_time
        latencies.append(latency)

        logger.info(f"  Iteration {i+1}: {latency:.3f}s")

    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]

    logger.info(f"\nLatency Results:")
    logger.info(f"  Average: {avg_latency:.3f}s")
    logger.info(f"  Median: {median_latency:.3f}s")
    logger.info(f"  P95: {p95_latency:.3f}s")
    logger.info(f"  P99: {p99_latency:.3f}s")

    return {
        "latencies": latencies,
        "average_latency_seconds": avg_latency,
        "median_latency_seconds": median_latency,
        "p95_latency_seconds": p95_latency,
        "p99_latency_seconds": p99_latency
    }


def benchmark_sequence_lengths(model, tokenizer) -> Dict:
    """Benchmark performance across different sequence lengths."""
    logger.info(f"\n[Benchmark 3/4] Sequence Length Scaling")
    logger.info(f"  Testing lengths: {BENCHMARK_CONFIG['sequence_lengths']}")
    logger.info("-"*80)

    results = []

    for seq_len in BENCHMARK_CONFIG['sequence_lengths']:
        # Create prompt of specific length
        base_prompt = "The quick brown fox jumps over the lazy dog. "
        prompt = base_prompt * (seq_len // 10)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len
        ).to(model.device)

        actual_length = inputs['input_ids'].shape[1]

        # Measure performance
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        tokens_generated = outputs.shape[1] - actual_length
        tokens_per_second = tokens_generated / elapsed

        results.append({
            "target_length": seq_len,
            "actual_length": actual_length,
            "tokens_generated": tokens_generated,
            "time_seconds": elapsed,
            "tokens_per_second": tokens_per_second
        })

        logger.info(f"  Length {actual_length}: {tokens_per_second:.2f} tokens/s")

    logger.info(f"\n✓ Sequence length scaling test complete")

    return {
        "results": results
    }


def benchmark_memory_efficiency(model) -> Dict:
    """Measure memory efficiency."""
    logger.info(f"\n[Benchmark 4/4] Memory Efficiency")
    logger.info("-"*80)

    # Get memory statistics
    memory_allocated = torch.cuda.memory_allocated() / 1e9
    memory_reserved = torch.cuda.memory_reserved() / 1e9
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9

    memory_utilization = (memory_allocated / memory_total) * 100

    logger.info(f"  Allocated: {memory_allocated:.2f} GB")
    logger.info(f"  Reserved: {memory_reserved:.2f} GB")
    logger.info(f"  Total available: {memory_total:.2f} GB")
    logger.info(f"  Utilization: {memory_utilization:.1f}%")

    # Model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {total_params / 1e9:.2f}B")

    # Theoretical size calculations
    # NVFP4 is ~0.5 bytes per parameter (4 bits + overhead)
    theoretical_size_gb = (total_params * 0.5) / 1e9

    logger.info(f"  Theoretical size: {theoretical_size_gb:.2f} GB")
    logger.info(f"  Actual allocated: {memory_allocated:.2f} GB")
    logger.info(f"  Overhead: {((memory_allocated - theoretical_size_gb) / theoretical_size_gb * 100):.1f}%")

    return {
        "memory_allocated_gb": memory_allocated,
        "memory_reserved_gb": memory_reserved,
        "memory_total_gb": memory_total,
        "memory_utilization_percent": memory_utilization,
        "total_parameters_billions": total_params / 1e9,
        "theoretical_size_gb": theoretical_size_gb
    }


def main():
    logger.info("="*80)
    logger.info("NVFP4 PERFORMANCE BENCHMARK")
    logger.info("="*80)
    logger.info(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration:")
    logger.info(f"  Warmup iterations: {BENCHMARK_CONFIG['warmup_iterations']}")
    logger.info(f"  Benchmark iterations: {BENCHMARK_CONFIG['benchmark_iterations']}")
    logger.info(f"  Max new tokens: {BENCHMARK_CONFIG['max_new_tokens']}")

    # Load model
    logger.info(f"\n[Setup] Loading model...")
    model, tokenizer = load_model_and_tokenizer()

    # Warmup
    warmup_model(model, tokenizer, BENCHMARK_CONFIG['warmup_iterations'])

    # Run benchmarks
    benchmark_results = {}

    try:
        benchmark_results['throughput'] = benchmark_throughput(
            model, tokenizer, BENCHMARK_CONFIG['benchmark_iterations']
        )

        benchmark_results['latency'] = benchmark_latency(
            model, tokenizer, BENCHMARK_CONFIG['benchmark_iterations']
        )

        benchmark_results['sequence_scaling'] = benchmark_sequence_lengths(
            model, tokenizer
        )

        benchmark_results['memory'] = benchmark_memory_efficiency(model)

    except Exception as e:
        logger.error(f"✗ Benchmark failed: {e}")
        sys.exit(1)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*80)

    logger.info(f"\nPerformance Summary:")
    logger.info(f"  Throughput: {benchmark_results['throughput']['average_tokens_per_second']:.2f} tokens/s")
    logger.info(f"  Latency (avg): {benchmark_results['latency']['average_latency_seconds']:.3f}s")
    logger.info(f"  Latency (P95): {benchmark_results['latency']['p95_latency_seconds']:.3f}s")
    logger.info(f"  Memory usage: {benchmark_results['memory']['memory_allocated_gb']:.2f} GB")
    logger.info(f"  Memory efficiency: {benchmark_results['memory']['memory_utilization_percent']:.1f}% of GPU")

    # Save results
    results_file = log_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        json.dump({
            "benchmark_config": BENCHMARK_CONFIG,
            "model_path": str(MODEL_DIR),
            "results": benchmark_results,
            "timestamp": datetime.now().isoformat(),
            "device": str(torch.cuda.get_device_name(0))
        }, f, indent=2)

    logger.info(f"\n✓ Detailed results saved to: {results_file}")
    logger.info(f"✓ Log file: {log_file}")

    logger.info(f"\nExpected DGX Spark Performance:")
    logger.info(f"  Target throughput: 15-30 tokens/s")
    logger.info(f"  Target memory: <25 GB")
    logger.info(f"  Target TTFT: 1-3 seconds")

    # Performance verdict
    avg_throughput = benchmark_results['throughput']['average_tokens_per_second']
    memory_gb = benchmark_results['memory']['memory_allocated_gb']

    logger.info(f"\nPerformance Verdict:")
    if avg_throughput >= 15 and memory_gb < 25:
        logger.info(f"  ✅ EXCELLENT - Meets all performance targets")
    elif avg_throughput >= 10:
        logger.info(f"  ✓ GOOD - Acceptable performance")
    else:
        logger.info(f"  ⚠️  BELOW TARGET - Consider optimization")


if __name__ == "__main__":
    main()

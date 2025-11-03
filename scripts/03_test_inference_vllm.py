#!/usr/bin/env python3
"""
NVFP4 Inference Testing Script (vLLM-based)
============================================

Tests the quantized NVFP4 model using vLLM with modelopt quantization support.
This is the correct way to load NVFP4 quantized models.

Requirements:
- vLLM 0.6.5 or later
- nvidia-modelopt

Usage:
    python scripts/03_test_inference_vllm.py

Prerequisites:
    - Quantized model created by 02_quantize_to_nvfp4.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"inference_test_vllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

# Test prompts covering different capabilities
TEST_PROMPTS = [
    {
        "name": "Technical Explanation",
        "prompt": "Explain how neural networks learn through backpropagation in simple terms.",
        "max_tokens": 256
    },
    {
        "name": "Code Generation",
        "prompt": "Write a Python function that implements binary search on a sorted list.",
        "max_tokens": 256
    },
    {
        "name": "Reasoning",
        "prompt": "If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is its average speed for the entire journey?",
        "max_tokens": 256
    },
    {
        "name": "Creative Writing",
        "prompt": "Write a short poem about artificial intelligence and the future.",
        "max_tokens": 256
    },
    {
        "name": "Instruction Following",
        "prompt": "List 5 benefits of using quantized models for AI inference. Format your answer as a numbered list.",
        "max_tokens": 256
    }
]


def load_quantized_model_vllm():
    """Load the NVFP4 quantized model using vLLM with modelopt quantization."""
    logger.info(f"Loading quantized model with vLLM from: {MODEL_DIR}")

    if not MODEL_DIR.exists():
        logger.error(f"✗ Quantized model not found at: {MODEL_DIR}")
        logger.error(f"  Run: ./docker-run.sh scripts/02_quantize_to_nvfp4.py first")
        sys.exit(1)

    # Verify required files exist
    required_files = ["config.json", "hf_quant_config.json"]
    missing_files = [f for f in required_files if not (MODEL_DIR / f).exists()]

    if missing_files:
        logger.error(f"✗ Missing required files: {missing_files}")
        sys.exit(1)

    try:
        from vllm import LLM, SamplingParams

        logger.info("  Initializing vLLM with modelopt quantization...")
        logger.info(f"  Model path: {MODEL_DIR}")
        logger.info(f"  Quantization: modelopt (NVFP4)")

        # Load the quantized model with vLLM
        # CRITICAL: Must specify quantization="modelopt" for NVFP4 models
        llm = LLM(
            model=str(MODEL_DIR),
            quantization="modelopt",
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=262144  # 256K context with YaRN RoPE scaling
        )

        logger.info(f"✓ Model loaded successfully with vLLM")

        return llm

    except ImportError as e:
        logger.error(f"✗ vLLM not installed: {e}")
        logger.error(f"  Install with: pip install vllm>=0.6.5")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def run_inference_tests(llm):
    """Run inference tests with various prompts using vLLM."""
    from vllm import SamplingParams

    logger.info(f"\nRunning inference tests on {len(TEST_PROMPTS)} prompts...")
    logger.info("="*80)

    results = []

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256
    )

    for i, test in enumerate(TEST_PROMPTS, 1):
        logger.info(f"\n[Test {i}/{len(TEST_PROMPTS)}] {test['name']}")
        logger.info("-"*80)
        logger.info(f"Prompt: {test['prompt']}")
        logger.info(f"\nGenerating response...")

        try:
            start_time = datetime.now()

            # Update sampling params for this test
            test_sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=test['max_tokens']
            )

            # Generate response
            outputs = llm.generate([test['prompt']], test_sampling_params)

            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()

            # Extract response
            output = outputs[0]
            response = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)
            tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0

            logger.info(f"\nResponse:\n{response}\n")
            logger.info(f"Metrics:")
            logger.info(f"  Time: {elapsed:.2f}s")
            logger.info(f"  Tokens: {tokens_generated}")
            logger.info(f"  Speed: {tokens_per_second:.2f} tokens/s")

            results.append({
                "test_name": test['name'],
                "prompt": test['prompt'],
                "response": response,
                "time_seconds": elapsed,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_second,
                "success": True
            })

        except Exception as e:
            logger.error(f"✗ Test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results.append({
                "test_name": test['name'],
                "prompt": test['prompt'],
                "error": str(e),
                "success": False
            })

    return results


def main():
    logger.info("="*80)
    logger.info("NVFP4 INFERENCE TESTING (vLLM + modelopt)")
    logger.info("="*80)
    logger.info(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load quantized model with vLLM
    logger.info(f"\n[1/3] Loading quantized model with vLLM...")
    llm = load_quantized_model_vllm()

    # Step 2: Run inference tests
    logger.info(f"\n[2/3] Running inference tests...")
    results = run_inference_tests(llm)

    # Step 3: Analyze results
    logger.info(f"\n[3/3] Analyzing results...")
    logger.info("="*80)

    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]

    logger.info(f"\nTest Results Summary:")
    logger.info(f"  Total tests: {len(results)}")
    logger.info(f"  Successful: {len(successful_tests)}")
    logger.info(f"  Failed: {len(failed_tests)}")

    if successful_tests:
        avg_time = sum(r['time_seconds'] for r in successful_tests) / len(successful_tests)
        avg_tokens = sum(r['tokens_generated'] for r in successful_tests) / len(successful_tests)
        avg_speed = sum(r['tokens_per_second'] for r in successful_tests) / len(successful_tests)

        logger.info(f"\nPerformance Metrics (Average):")
        logger.info(f"  Time per test: {avg_time:.2f}s")
        logger.info(f"  Tokens per test: {avg_tokens:.1f}")
        logger.info(f"  Speed: {avg_speed:.2f} tokens/s")

    # Save results
    results_dir = Path(__file__).parent.parent / "logs"
    results_file = results_dir / f"inference_results_vllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        json.dump({
            "summary": {
                "total_tests": len(results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "avg_time_seconds": avg_time if successful_tests else 0,
                "avg_tokens_generated": avg_tokens if successful_tests else 0,
                "avg_tokens_per_second": avg_speed if successful_tests else 0,
            },
            "test_results": results,
            "timestamp": datetime.now().isoformat(),
            "inference_framework": "vLLM with modelopt quantization"
        }, f, indent=2)

    logger.info(f"\n✓ Results saved to: {results_file}")

    # Final verdict
    logger.info("\n" + "="*80)
    logger.info("INFERENCE TESTING COMPLETE")
    logger.info("="*80)

    if len(successful_tests) == len(results):
        logger.info(f"\n✅ ALL TESTS PASSED!")
        logger.info(f"\nThe quantized NVFP4 model is working correctly with vLLM.")
        logger.info(f"Average speed: {avg_speed:.2f} tokens/s")
    elif successful_tests:
        logger.warning(f"\n⚠️  PARTIAL SUCCESS")
        logger.warning(f"{len(failed_tests)} test(s) failed")
        logger.warning(f"Check log file for details: {log_file}")
    else:
        logger.error(f"\n❌ ALL TESTS FAILED")
        logger.error(f"Check log file for details: {log_file}")
        sys.exit(1)

    logger.info(f"\nLog file: {log_file}")


if __name__ == "__main__":
    main()

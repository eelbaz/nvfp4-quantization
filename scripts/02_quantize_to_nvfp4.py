#!/usr/bin/env python3
"""
NVFP4 Quantization Script for Qwen3-VLTO-32B-Instruct
=====================================================

Quantizes the Qwen3-VLTO-32B-Instruct model to NVFP4 format using
NVIDIA TensorRT Model Optimizer for hardware-optimized inference on DGX Spark.

NVFP4 uses a two-level scaling strategy:
- E4M3 FP8 scaling factors per 16-value block
- Global FP32 tensor scale
- Reduces quantization error while maintaining model accuracy

Expected memory savings: ~70% (32GB BF16 → ~10GB NVFP4)

Usage:
    uv run scripts/02_quantize_to_nvfp4.py

Prerequisites:
    - Run 01_prepare_calibration_data.py first
    - Model downloaded to HF cache
    - DGX Spark with GB10 GPU
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import modelopt.torch.quantization as mtq
import modelopt.torch.export as mte

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"nvfp4_quantization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
HF_CACHE_DIR = Path(config['hf_cache_dir']).expanduser()
MODEL_NAME = config['model_name']
CALIB_DATA_PATH = Path(__file__).parent.parent / "calibration-data" / "calibration.json"
OUTPUT_DIR = Path(__file__).parent.parent / config['output_dir']


def load_calibration_data() -> List[Dict]:
    """Load calibration data prepared by previous script."""
    logger.info(f"Loading calibration data from: {CALIB_DATA_PATH}")

    if not CALIB_DATA_PATH.exists():
        logger.error(f"✗ Calibration data not found!")
        logger.error(f"  Run: uv run scripts/01_prepare_calibration_data.py first")
        sys.exit(1)

    with open(CALIB_DATA_PATH, 'r') as f:
        data = json.load(f)

    logger.info(f"✓ Loaded {len(data)} calibration samples")
    return data


def create_forward_loop(model, tokenizer, calibration_data: List[Dict]):
    """
    Create forward loop function for calibration.

    This function runs the model on calibration data to collect
    activation statistics for optimal quantization.
    """
    def forward_loop():
        logger.info("  Starting calibration forward pass...")
        model.eval()

        with torch.no_grad():
            for idx, sample in enumerate(calibration_data):
                # Tokenize input
                inputs = tokenizer(
                    sample['text'],
                    return_tensors="pt",
                    max_length=config['calibration_max_length'],
                    truncation=True
                ).to(model.device)

                # Forward pass (no gradient computation)
                model(**inputs)

                # Progress logging
                if (idx + 1) % 50 == 0:
                    logger.info(f"    Calibrated {idx + 1}/{len(calibration_data)} samples...")

        logger.info(f"  ✓ Calibration forward pass complete")

    return forward_loop


def main():
    logger.info("="*80)
    logger.info("NVFP4 QUANTIZATION PROCESS")
    logger.info("="*80)
    logger.info(f"\nTarget: {MODEL_NAME}")
    logger.info(f"Quantization: NVFP4 (4-bit floating point)")
    logger.info(f"Hardware: DGX Spark (GB10 Blackwell)")
    logger.info(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Load calibration data
    logger.info(f"\n[1/6] Loading calibration data...")
    calibration_data = load_calibration_data()

    # Step 2: Load model
    logger.info(f"\n[2/6] Loading model from Hugging Face cache...")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Cache: {HF_CACHE_DIR}")
    logger.info(f"  Dtype: {config['torch_dtype']}")
    logger.info(f"  This may take 5-10 minutes...")

    try:
        # Load model in bfloat16
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map[config['torch_dtype']]

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=config['device_map'],
            cache_dir=str(HF_CACHE_DIR),
            trust_remote_code=config['trust_remote_code']
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(HF_CACHE_DIR),
            trust_remote_code=config['trust_remote_code']
        )

        # Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        memory_allocated = torch.cuda.memory_allocated() / 1e9

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Parameters: {total_params / 1e9:.2f}B")
        logger.info(f"  Memory footprint: {memory_allocated:.2f} GB")
        logger.info(f"  Device: {next(model.parameters()).device}")

    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error(f"\nTroubleshooting:")
        logger.error(f"  1. Verify model is downloaded to HF cache")
        logger.error(f"  2. Check GPU memory availability")
        logger.error(f"  3. Ensure CUDA is properly configured")
        sys.exit(1)

    # Step 3: Create calibration forward loop
    logger.info(f"\n[3/6] Preparing calibration function...")
    forward_loop = create_forward_loop(model, tokenizer, calibration_data)
    logger.info(f"✓ Calibration function ready")

    # Step 4: Quantize to NVFP4
    logger.info(f"\n[4/6] Quantizing model to NVFP4...")
    logger.info(f"  This will take 30-60 minutes depending on hardware...")
    logger.info(f"  Progress updates will appear during calibration...")

    try:
        # Apply NVFP4 quantization
        # NVFP4_DEFAULT_CFG configures:
        # - 4-bit floating point representation
        # - Two-level scaling (per-block E4M3 + global FP32)
        # - Optimized for Blackwell Tensor Cores
        quantized_model = mtq.quantize(
            model,
            mtq.NVFP4_DEFAULT_CFG,
            forward_loop
        )

        logger.info(f"\n✓ Quantization complete!")

        # Print quantization summary
        logger.info(f"\nQuantization Summary:")
        logger.info("-" * 80)
        mtq.print_quant_summary(quantized_model)
        logger.info("-" * 80)

        # Memory comparison
        quantized_memory = torch.cuda.memory_allocated() / 1e9
        memory_reduction = ((memory_allocated - quantized_memory) / memory_allocated) * 100

        logger.info(f"\nMemory Statistics:")
        logger.info(f"  Original (BF16): {memory_allocated:.2f} GB")
        logger.info(f"  Quantized (NVFP4): {quantized_memory:.2f} GB")
        logger.info(f"  Reduction: {memory_reduction:.1f}%")

    except Exception as e:
        logger.error(f"✗ Quantization failed: {e}")
        logger.error(f"\nTroubleshooting:")
        logger.error(f"  1. Check GPU memory (nvidia-smi)")
        logger.error(f"  2. Verify calibration data is valid")
        logger.error(f"  3. Try reducing calibration_samples in config")
        sys.exit(1)

    # Step 5: Export quantized model
    logger.info(f"\n[5/6] Exporting quantized model...")
    logger.info(f"  Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Move all tensors to cuda:0 before export (fixes mixed device issue)
        logger.info(f"  Moving all tensors to cuda:0 for export...")
        quantized_model = quantized_model.to('cuda:0')
        logger.info(f"  ✓ All tensors on cuda:0")

        with torch.inference_mode():
            # Export in HuggingFace format for compatibility
            mte.export_hf_checkpoint(
                quantized_model,
                export_dir=str(OUTPUT_DIR),
                dtype=torch.float16  # Export format for weights
            )

        # Save tokenizer
        tokenizer.save_pretrained(str(OUTPUT_DIR))

        # Verify exported files
        exported_files = list(OUTPUT_DIR.glob("*"))
        logger.info(f"✓ Model exported successfully")
        logger.info(f"  Files created: {len(exported_files)}")

        # List key files
        logger.info(f"\n  Key files:")
        for pattern in ["config.json", "hf_quant_config.json", "*.safetensors", "tokenizer*"]:
            matching = list(OUTPUT_DIR.glob(pattern))
            for f in matching:
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"    - {f.name} ({size_mb:.1f} MB)")

    except Exception as e:
        logger.error(f"✗ Export failed: {e}")
        sys.exit(1)

    # Step 6: Save metadata
    logger.info(f"\n[6/6] Saving quantization metadata...")

    metadata = {
        "model_name": MODEL_NAME,
        "quantization_format": "NVFP4",
        "calibration_samples": len(calibration_data),
        "original_memory_gb": f"{memory_allocated:.2f}",
        "quantized_memory_gb": f"{quantized_memory:.2f}",
        "memory_reduction_percent": f"{memory_reduction:.1f}",
        "total_parameters_billions": f"{total_params / 1e9:.2f}",
        "target_device": config['target_device'],
        "export_path": str(OUTPUT_DIR),
        "quantization_config": config,
        "created_at": datetime.now().isoformat(),
    }

    metadata_file = OUTPUT_DIR / "quantization_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Metadata saved: {metadata_file}")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("QUANTIZATION COMPLETE!")
    logger.info("="*80)

    logger.info(f"\nModel Information:")
    logger.info(f"  ✓ Model: {MODEL_NAME}")
    logger.info(f"  ✓ Format: NVFP4 (4-bit floating point)")
    logger.info(f"  ✓ Parameters: {total_params / 1e9:.2f}B")
    logger.info(f"  ✓ Memory: {quantized_memory:.2f} GB ({memory_reduction:.1f}% reduction)")

    logger.info(f"\nOutput:")
    logger.info(f"  ✓ Location: {OUTPUT_DIR}")
    logger.info(f"  ✓ Metadata: {metadata_file}")
    logger.info(f"  ✓ Log file: {log_file}")

    logger.info(f"\nNext Steps:")
    logger.info(f"  1. Verify quantization: uv run scripts/03_test_inference.py")
    logger.info(f"  2. Benchmark performance: uv run scripts/04_benchmark.py")
    logger.info(f"  3. Deploy with vLLM or TensorRT-LLM")

    logger.info(f"\nVerification Checklist:")
    logger.info(f"  □ Check hf_quant_config.json contains NVFP4 configuration")
    logger.info(f"  □ Verify .safetensors files exist and are smaller than original")
    logger.info(f"  □ Run inference test to validate model quality")
    logger.info(f"  □ Compare responses with original model (if available)")


if __name__ == "__main__":
    main()

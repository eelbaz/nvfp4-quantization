#!/usr/bin/env python3
"""
Calibration Data Preparation for NVFP4 Quantization
====================================================

Prepares a calibration dataset for post-training quantization of
Qwen3-VLTO-32B-Instruct to NVFP4 format.

This script:
1. Loads the model tokenizer from HF cache
2. Downloads calibration data (C4 dataset)
3. Tokenizes and filters samples
4. Saves calibration data for quantization process

Usage:
    uv run scripts/01_prepare_calibration_data.py
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"calibration_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

# Expand HF cache directory
HF_CACHE_DIR = Path(config['hf_cache_dir']).expanduser()
MODEL_NAME = config['model_name']
NUM_SAMPLES = config['calibration_samples']
MAX_LENGTH = config['calibration_max_length']
MIN_LENGTH = config['calibration_min_length']


def main():
    logger.info("="*80)
    logger.info("CALIBRATION DATA PREPARATION")
    logger.info("="*80)

    # Step 1: Load tokenizer from HF cache
    logger.info(f"\n[1/4] Loading tokenizer from Hugging Face cache...")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Cache directory: {HF_CACHE_DIR}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=str(HF_CACHE_DIR),
            trust_remote_code=config['trust_remote_code']
        )
        logger.info(f"✓ Tokenizer loaded successfully")
        logger.info(f"  Vocab size: {len(tokenizer)}")
    except Exception as e:
        logger.error(f"✗ Failed to load tokenizer: {e}")
        sys.exit(1)

    # Step 2: Load calibration dataset
    logger.info(f"\n[2/4] Loading calibration dataset...")
    logger.info(f"  Dataset: {config['calibration_dataset']}")
    logger.info(f"  Config: {config['calibration_dataset_config']}")
    logger.info(f"  Split: {config['calibration_dataset_split']}")

    try:
        dataset = load_dataset(
            config['calibration_dataset'],
            config['calibration_dataset_config'],
            split=config['calibration_dataset_split'],
            streaming=True,
            cache_dir=str(HF_CACHE_DIR)
        )
        logger.info(f"✓ Dataset loaded successfully (streaming mode)")
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {e}")
        sys.exit(1)

    # Step 3: Process and filter samples
    logger.info(f"\n[3/4] Processing calibration samples...")
    logger.info(f"  Target samples: {NUM_SAMPLES}")
    logger.info(f"  Max length: {MAX_LENGTH} tokens")
    logger.info(f"  Min length: {MIN_LENGTH} tokens")

    calibration_samples: List[Dict] = []
    processed_count = 0
    valid_count = 0

    try:
        for idx, example in enumerate(dataset):
            processed_count += 1

            # Break if we have enough samples
            if valid_count >= NUM_SAMPLES:
                break

            # Tokenize text
            text = example['text']
            tokens = tokenizer.encode(
                text,
                max_length=MAX_LENGTH,
                truncation=True
            )

            # Filter: must be longer than minimum length
            if len(tokens) > MIN_LENGTH:
                calibration_samples.append({
                    'text': tokenizer.decode(tokens),
                    'input_ids': tokens,
                    'length': len(tokens)
                })
                valid_count += 1

                # Progress logging
                if valid_count % 50 == 0:
                    logger.info(f"  Progress: {valid_count}/{NUM_SAMPLES} valid samples collected")

            # Safety limit to prevent infinite loop
            if processed_count > NUM_SAMPLES * 10:
                logger.warning(f"  Processed {processed_count} samples but only found {valid_count} valid ones")
                logger.warning(f"  Stopping to prevent excessive processing")
                break

    except Exception as e:
        logger.error(f"✗ Error during sample processing: {e}")
        sys.exit(1)

    # Calculate statistics
    avg_length = sum(s['length'] for s in calibration_samples) / len(calibration_samples)
    min_sample_length = min(s['length'] for s in calibration_samples)
    max_sample_length = max(s['length'] for s in calibration_samples)

    logger.info(f"\n✓ Sample processing complete!")
    logger.info(f"  Total processed: {processed_count}")
    logger.info(f"  Valid samples: {valid_count}")
    logger.info(f"  Average length: {avg_length:.1f} tokens")
    logger.info(f"  Length range: {min_sample_length} - {max_sample_length} tokens")

    # Step 4: Save calibration data
    logger.info(f"\n[4/4] Saving calibration data...")

    output_dir = Path(__file__).parent.parent / "calibration-data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "calibration.json"

    try:
        with open(output_file, 'w') as f:
            json.dump(calibration_samples, f, indent=2)

        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✓ Calibration data saved successfully")
        logger.info(f"  Location: {output_file}")
        logger.info(f"  File size: {file_size:.2f} MB")

        # Save metadata
        metadata = {
            "model_name": MODEL_NAME,
            "num_samples": len(calibration_samples),
            "avg_length": avg_length,
            "min_length": min_sample_length,
            "max_length": max_sample_length,
            "dataset": config['calibration_dataset'],
            "created_at": datetime.now().isoformat(),
            "config": config
        }

        metadata_file = output_dir / "calibration_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"  Metadata: {metadata_file}")

    except Exception as e:
        logger.error(f"✗ Failed to save calibration data: {e}")
        sys.exit(1)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("CALIBRATION DATA PREPARATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nSummary:")
    logger.info(f"  ✓ {valid_count} calibration samples prepared")
    logger.info(f"  ✓ Average length: {avg_length:.1f} tokens")
    logger.info(f"  ✓ Data saved to: {output_file}")
    logger.info(f"\nNext step:")
    logger.info(f"  Run: uv run scripts/02_quantize_to_nvfp4.py")
    logger.info(f"\nLog file: {log_file}")


if __name__ == "__main__":
    main()

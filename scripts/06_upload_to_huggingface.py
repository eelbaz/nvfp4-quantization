#!/usr/bin/env python3
"""
Upload NVFP4 Quantized Model to HuggingFace
===========================================

Uploads the quantized model to HuggingFace Hub with the model card.

Usage:
    python scripts/06_upload_to_huggingface.py

Prerequisites:
    - Quantized model exists in quantized-output/
    - MODEL_CARD.md exists
    - HuggingFace authentication configured (huggingface-cli login)
"""

import json
import sys
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"huggingface_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
QUANTIZED_MODEL_DIR = Path(__file__).parent.parent / config['output_dir']
MODEL_CARD_PATH = Path(__file__).parent.parent / "MODEL_CARD.md"
REPO_ID = "Ex0bit/Qwen3-VLTO-32B-Instruct-NVFP4"


def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("Checking prerequisites...")

    # Check if quantized model exists
    if not QUANTIZED_MODEL_DIR.exists():
        logger.error(f"Quantized model directory not found: {QUANTIZED_MODEL_DIR}")
        logger.error("Run: ./docker-run.sh scripts/02_quantize_to_nvfp4.py first")
        return False

    # Check for model files
    model_files = list(QUANTIZED_MODEL_DIR.glob("*.safetensors"))
    if not model_files:
        logger.error(f"No safetensors files found in {QUANTIZED_MODEL_DIR}")
        return False

    logger.info(f"Found {len(model_files)} safetensors files")

    # Check if MODEL_CARD.md exists
    if not MODEL_CARD_PATH.exists():
        logger.error(f"MODEL_CARD.md not found: {MODEL_CARD_PATH}")
        return False

    logger.info(f"MODEL_CARD.md found at {MODEL_CARD_PATH}")

    # Check HuggingFace authentication
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        logger.info(f"Authenticated as: {user_info['name']}")
    except Exception as e:
        logger.error(f"HuggingFace authentication failed: {e}")
        logger.error("Run: huggingface-cli login")
        return False

    return True


def prepare_upload_directory():
    """Prepare a temporary directory for upload."""
    logger.info("Preparing upload directory...")

    # Copy MODEL_CARD.md to quantized model directory as README.md
    readme_path = QUANTIZED_MODEL_DIR / "README.md"
    shutil.copy2(MODEL_CARD_PATH, readme_path)
    logger.info(f"Copied MODEL_CARD.md to {readme_path}")

    return True


def upload_model():
    """Upload the model to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo

    logger.info("="*80)
    logger.info("UPLOADING TO HUGGINGFACE")
    logger.info("="*80)
    logger.info(f"\nRepository: {REPO_ID}")
    logger.info(f"Model directory: {QUANTIZED_MODEL_DIR}")

    try:
        # Initialize API
        api = HfApi()

        # Create repository (will skip if already exists)
        logger.info(f"\nCreating repository: {REPO_ID}")
        try:
            create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
            logger.info(f"Repository created/verified: https://huggingface.co/{REPO_ID}")
        except Exception as e:
            logger.warning(f"Repository may already exist: {e}")

        # Upload entire folder
        logger.info(f"\nUploading model files...")
        logger.info(f"This may take 15-30 minutes depending on network speed...")

        api.upload_folder(
            folder_path=str(QUANTIZED_MODEL_DIR),
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload NVFP4 quantized Qwen3-VLTO-32B-Instruct model"
        )

        logger.info("="*80)
        logger.info("UPLOAD COMPLETE")
        logger.info("="*80)
        logger.info(f"\nModel uploaded successfully!")
        logger.info(f"View at: https://huggingface.co/{REPO_ID}")
        logger.info(f"\nYou can now load the model with:")
        logger.info(f"  from vllm import LLM")
        logger.info(f"  llm = LLM(model='{REPO_ID}', quantization='modelopt')")

        return True

    except Exception as e:
        logger.error(f"\nUpload failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    logger.info("="*80)
    logger.info("HUGGINGFACE UPLOAD SCRIPT")
    logger.info("="*80)
    logger.info(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not check_prerequisites():
        logger.error("\nPrerequisites not met. Exiting.")
        sys.exit(1)

    # Prepare upload directory
    if not prepare_upload_directory():
        logger.error("\nFailed to prepare upload directory. Exiting.")
        sys.exit(1)

    # Upload model
    if not upload_model():
        logger.error("\nUpload failed. Exiting.")
        sys.exit(1)

    logger.info(f"\nLog file: {log_file}")
    logger.info("\nAll done!")


if __name__ == "__main__":
    main()

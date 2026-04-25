#!/usr/bin/env python3
"""
Setup Ollama Models Management

Pulls and manages models from YAML configuration.
Cloud models (e.g. ones with :cloud) are handled properly by bypassing the local pull.
"""

import sys
import argparse
import logging
import subprocess
import asyncio
from pathlib import Path
from typing import List, Dict
import yaml

from llm_client import create_client


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG_FILE = 'model_config.yaml'


# ============================================================================
# Logging
# ============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: Path) -> List[str]:
    """Load model list from YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract model list
    models = config.get('ollama_models', [])

    if not models:
        raise ValueError(f"No models found in config: {config_path}")

    return models


# ============================================================================
# Ollama Operations
# ============================================================================

def check_ollama_installed() -> bool:
    """Check if Ollama CLI is installed."""
    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_installed_models() -> List[str]:
    """Get list of currently installed Ollama models."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return []

        # Parse output (skip header line)
        lines = result.stdout.strip().split('\n')[1:]
        models = []
        for line in lines:
            if line.strip():
                # Model name is first column
                model_name = line.split()[0]
                models.append(model_name)

        return models

    except Exception:
        return []


def pull_model(model_name: str, logger: logging.Logger) -> bool:
    """Pull a single Ollama model."""
    if "cloud" in model_name.lower():
        logger.info(f"  ✓ {model_name} is a cloud model, skipping local pull.")
        return True

    logger.info(f"Pulling model: {model_name}")

    try:
        result = subprocess.run(
            ['ollama', 'pull', model_name],
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            logger.info(f"  ✓ Successfully pulled: {model_name}")
            return True
        else:
            logger.error(f"  ✗ Failed to pull: {model_name}")
            return False

    except Exception as e:
        logger.error(f"  ✗ Error pulling {model_name}: {e}")
        return False


# ============================================================================
# Main Operations
# ============================================================================

async def test_cloud_model(model_name: str, logger: logging.Logger) -> bool:
    """Test a cloud model by making a short request."""
    try:
        client = create_client(
            model_name=model_name,
            temperature=0.7,
            max_tokens=64
        )
        messages = [
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Respond with 'Hello'."}
        ]
        response = await client.chat(messages=messages)
        if response and response.content:
            logger.info(f"  ✓ Successfully tested cloud model: {model_name}")
            return True
        else:
            logger.error(f"  ✗ Empty response from cloud model: {model_name}")
            return False
    except Exception as e:
        logger.error(f"  ✗ Error testing cloud model {model_name}: {e}")
        return False

def pull_models_from_config(
    config_path: Path,
    logger: logging.Logger,
    dry_run: bool = False,
    skip_existing: bool = True
) -> Dict[str, List[str]]:
    """
    Pull all models from a config file.
    """
    models = load_config(config_path)
    logger.info(f"Found {len(models)} models in config: {config_path.name}")

    installed = get_installed_models() if skip_existing else []

    results = {
        'success': [],
        'failed': [],
        'skipped': [],
    }

    for i, model in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] Processing: {model}")

        if "cloud" in model.lower():
            logger.info(f"  → Cloud model detected. Testing connection...")
            if dry_run:
                logger.info(f"  → Would test connection (dry-run)")
                continue
                
            success = asyncio.run(test_cloud_model(model, logger))
            if success:
                results['success'].append(model)
            else:
                results['failed'].append(model)
            continue

        # Skip if already installed
        if model in installed:
            logger.info(f"  → Already installed, skipping")
            results['skipped'].append(model)
            continue

        if dry_run:
            logger.info(f"  → Would pull (dry-run)")
            continue

        # Pull the model
        success = pull_model(model, logger)

        if success:
            results['success'].append(model)
        else:
            results['failed'].append(model)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ollama Model Management for Project"
    )

    parser.add_argument('--config-file', type=str, default=DEFAULT_CONFIG_FILE,
                       help='Path to custom config YAML file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without doing it')
    parser.add_argument('--force', action='store_true',
                       help='Pull even if model is already installed')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("=" * 60)
    logger.info("Ollama Model Management")
    logger.info("=" * 60)

    # Check Ollama installation
    if not check_ollama_installed():
        logger.error("Ollama is not installed or not in PATH")
        logger.error("Install from: https://ollama.ai/")
        sys.exit(1)

    logger.info("Ollama CLI: OK")

    script_dir = Path(__file__).parent
    config_path = Path(args.config_file)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    logger.info(f"Using config: {config_path}")

    results = pull_models_from_config(
        config_path,
        logger,
        dry_run=args.dry_run,
        skip_existing=not args.force
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("(Dry run - no models were actually pulled)")

    logger.info(f"Success: {len(results['success'])}")
    logger.info(f"Failed:  {len(results['failed'])}")
    logger.info(f"Skipped: {len(results['skipped'])}")

    if results['failed']:
        logger.warning("\nFailed models:")
        for m in results['failed']:
            logger.warning(f"  - {m}")

    sys.exit(0 if not results['failed'] else 1)


if __name__ == "__main__":
    main()

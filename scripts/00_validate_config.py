#!/usr/bin/env python3
"""Validate experiment_config.yaml. Fails loudly if required fields are null."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, resolve_config, validate_config


def main() -> None:
    cfg = load_config()
    cfg = resolve_config(cfg)
    validate_config(cfg)
    print("Config validation passed.")


if __name__ == "__main__":
    main()

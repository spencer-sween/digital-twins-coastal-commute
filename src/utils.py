"""Utility helpers shared across pipeline stages."""
import json
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Union


def ensure_dirs(*dirs) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_resolved_config(cfg: dict, out_dir: Union[str, Path]) -> None:
    path = Path(out_dir) / "config_resolved.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[{timestamp()}] {msg}")

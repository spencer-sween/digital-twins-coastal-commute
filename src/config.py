"""Load and validate experiment configuration."""
import os
import yaml
from pathlib import Path
from typing import Optional


def load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "experiment_config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_config(cfg: dict) -> dict:
    """Apply env-var overrides and fill derived fields."""
    env_model = os.environ.get("ANTHROPIC_MODEL")
    if env_model:
        cfg["anthropic"]["model"] = env_model
    return cfg


REQUIRED_FIELDS = [
    ("experiment", "n_per_job_type"),
    ("routes", "freeway_distance_miles"),
    ("routes", "coastal_distance_miles"),
    ("routes", "base_times_freeway"),
    ("routes", "base_times_coastal"),
]


def validate_config(cfg: dict) -> None:
    errors = []
    for section, key in REQUIRED_FIELDS:
        val = cfg.get(section, {}).get(key)
        if val is None:
            errors.append(f"configs/experiment_config.yaml: [{section}].{key} is null")
    # Check nested time dicts
    for route in ("base_times_freeway", "base_times_coastal"):
        times = cfg.get("routes", {}).get(route, {})
        for t in cfg.get("randomization", {}).get("release_times", []):
            if times.get(t) is None:
                errors.append(f"[routes].{route}[{t}] is null")
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

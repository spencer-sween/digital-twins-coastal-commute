#!/usr/bin/env python3
"""Phase 1: Generate experimental design with bounded heavy-tailed jitter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, resolve_config, validate_config
from src.simulate_design import simulate_design
from src.utils import ensure_dirs, log


def main() -> None:
    cfg = load_config()
    cfg = resolve_config(cfg)
    validate_config(cfg)

    log("Simulating experimental design...")
    design = simulate_design(cfg)

    ensure_dirs("outputs/data")
    design.to_csv("outputs/data/design_pre_llm.csv", index=False)
    log(f"Design saved: {len(design)} observations across {design['job_type'].nunique()} job types.")
    print(design[["obs_id", "job_type", "release_time", "fatigue_state", "weather",
                   "freeway_time_minutes", "coastal_time_minutes", "delta_time_coastal_minus_freeway"]].head(10).to_string())


if __name__ == "__main__":
    main()

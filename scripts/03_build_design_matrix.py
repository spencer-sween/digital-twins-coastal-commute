#!/usr/bin/env python3
"""Phase 3: Build estimation-ready design matrix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, resolve_config
from src.build_design_matrix import build_design_matrix
from src.utils import ensure_dirs, log
import pandas as pd


def main() -> None:
    cfg = load_config()
    cfg = resolve_config(cfg)

    panel = pd.read_csv("outputs/data/person_level_panel.csv")
    log(f"Building design matrix from {len(panel)} observations...")

    dm = build_design_matrix(panel, cfg)
    W_df = dm["W_df"].copy()
    W_df.insert(0, "obs_id", dm["df"]["obs_id"].values)

    ensure_dirs("outputs/data", "outputs/estimates")
    W_df.to_csv("outputs/data/estimation_design_matrix.csv", index=False)
    log(f"Design matrix: {W_df.shape[1]-1} features × {len(W_df)} observations.")
    log(f"Feature names: {dm['feature_names'][:5]}...")


if __name__ == "__main__":
    main()

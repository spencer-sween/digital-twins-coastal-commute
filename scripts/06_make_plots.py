#!/usr/bin/env python3
"""Phase 7: Generate all 18 required plots."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, resolve_config
from src.build_design_matrix import build_design_matrix
from src.estimate_logit import fit_logit
from src.flm_debias import cross_fit_logit, compute_influence_functions
from src.targets import compute_targets
from src.plots import make_all_plots
from src.utils import ensure_dirs, log


def main() -> None:
    cfg = load_config()
    cfg = resolve_config(cfg)

    panel = pd.read_csv("outputs/data/person_level_panel.csv")
    dm = build_design_matrix(panel, cfg)
    W, Y, D = dm["W"], dm["Y"], dm["D"]
    A, B = dm["A"].values, dm["B"].values
    n_A = A.shape[1]

    log("Fitting model for plots...")
    fit = fit_logit(W, Y, ridge_alpha=cfg.get("estimation", {}).get("ridge_alpha", 0.01))
    cross_fit = cross_fit_logit(W, Y, cfg)
    flm = compute_influence_functions(cross_fit, W, Y, A, B, D, n_A)
    target_result = compute_targets(panel, dm, fit, flm, cfg)

    ensure_dirs("outputs/figures")
    log("Generating plots...")
    make_all_plots(panel, dm, fit, target_result, "outputs/figures")
    log("All plots saved.")


if __name__ == "__main__":
    main()

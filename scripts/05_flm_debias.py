#!/usr/bin/env python3
"""Phase 5: FLM-style debiasing with K-fold cross-fitting."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, resolve_config
from src.build_design_matrix import build_design_matrix
from src.estimate_logit import fit_logit, invert_hessian
from src.flm_debias import cross_fit_logit, compute_influence_functions
from src.utils import ensure_dirs, log


def main() -> None:
    cfg = load_config()
    cfg = resolve_config(cfg)

    panel = pd.read_csv("outputs/data/person_level_panel.csv")
    dm = build_design_matrix(panel, cfg)
    W, Y, D = dm["W"], dm["Y"], dm["D"]
    A, B = dm["A"].values, dm["B"].values
    n_A = A.shape[1]

    log("Cross-fitting logistic regression...")
    cross_fit = cross_fit_logit(W, Y, cfg)
    log(f"Cross-fitting done. {cross_fit['n_folds']} folds.")

    log("Computing influence functions...")
    flm = compute_influence_functions(cross_fit, W, Y, A, B, D, n_A)

    ensure_dirs("outputs/estimates")

    # Influence functions
    if_df = dm["df"][["obs_id"]].copy()
    if_df["phi_ame"] = flm["phi_ame"]
    if_df["b_i"] = flm["b_i"]
    if_df["me_i"] = flm["me_i"]
    if_df.to_csv("outputs/estimates/influence_functions.csv", index=False)

    # Summary
    summary = pd.DataFrame([{
        "target": "AME (naive)",
        "estimate": flm["tau_ame"],
        "se": flm["se_ame"],
        "ci_lower": flm["tau_ame"] - 1.96 * flm["se_ame"],
        "ci_upper": flm["tau_ame"] + 1.96 * flm["se_ame"],
    }, {
        "target": "AME (debiased)",
        "estimate": flm["tau_ame_debiased"],
        "se": flm["se_ame"],
        "ci_lower": flm["tau_ame_debiased"] - 1.96 * flm["se_ame"],
        "ci_upper": flm["tau_ame_debiased"] + 1.96 * flm["se_ame"],
    }])
    summary.to_csv("outputs/estimates/target_parameter_summary.csv", index=False)
    log(f"AME (debiased) = {flm['tau_ame_debiased']:.4f}  SE = {flm['se_ame']:.4f}")


if __name__ == "__main__":
    main()

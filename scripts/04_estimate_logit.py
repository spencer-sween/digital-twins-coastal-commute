#!/usr/bin/env python3
"""Phase 4: Ridge-regularized logistic regression with Hessian diagnostics."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, resolve_config
from src.build_design_matrix import build_design_matrix
from src.estimate_logit import fit_logit, hessian_to_long, invert_hessian
from src.utils import ensure_dirs, log


def main() -> None:
    cfg = load_config()
    cfg = resolve_config(cfg)

    panel = pd.read_csv("outputs/data/person_level_panel.csv")
    dm = build_design_matrix(panel, cfg)
    W, Y = dm["W"], dm["Y"]
    ridge_alpha = cfg.get("estimation", {}).get("ridge_alpha", 0.01)

    log(f"Fitting logistic regression: {W.shape[1]} features, {len(Y)} obs, ridge_alpha={ridge_alpha}...")
    fit = fit_logit(W, Y, ridge_alpha=ridge_alpha)

    log(f"Condition number: {fit['condition_number']:.2f}")
    log(f"Fitted coastal share: {fit['p'].mean():.4f}  (observed: {Y.mean():.4f})")

    ensure_dirs("outputs/estimates")

    # Coefficients
    coef_df = pd.DataFrame({
        "feature": dm["feature_names"],
        "coefficient": fit["theta"],
    })
    coef_df.to_csv("outputs/estimates/logit_coefficients.csv", index=False)

    # Fitted values
    fit_df = dm["df"][["obs_id"]].copy()
    fit_df["p_hat"] = fit["p"]
    fit_df["choose_coastal"] = Y
    fit_df.to_csv("outputs/estimates/logit_fitted_values.csv", index=False)

    # Scores
    score_df = pd.DataFrame(fit["scores"], columns=[f"score_{f}" for f in dm["feature_names"]])
    score_df.insert(0, "obs_id", dm["df"]["obs_id"].values)
    score_df.to_csv("outputs/estimates/logit_scores.csv", index=False)

    # Hessian
    H_long = hessian_to_long(fit["H"], dm["feature_names"])
    H_long.to_csv("outputs/estimates/hessian_long.csv", index=False)
    pd.DataFrame(fit["H"], index=dm["feature_names"], columns=dm["feature_names"]).to_csv(
        "outputs/estimates/hessian_matrix.csv"
    )

    # Eigenvalues
    eig_df = pd.DataFrame({"eigenvalue": fit["eigenvalues"]})
    eig_df.to_csv("outputs/estimates/hessian_eigenvalues.csv", index=False)

    log(f"Results saved to outputs/estimates/")


if __name__ == "__main__":
    main()

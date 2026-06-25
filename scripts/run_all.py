#!/usr/bin/env python3
"""
End-to-end pipeline orchestrator.

Usage:
  python scripts/run_all.py --dry-run          # seeded fake choices, no API calls
  python scripts/run_all.py --api-run          # real API calls (costs ~$6.50 for 5000 sims)
  python scripts/run_all.py --api-run --skip-report
  python scripts/run_all.py --dry-run --n 10  # override n_per_job_type for quick test
"""
import argparse
import sys
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, resolve_config, validate_config
from src.simulate_design import simulate_design
from src.run_experiment import run_experiment
from src.build_design_matrix import build_design_matrix
from src.estimate_logit import fit_logit, hessian_to_long
from src.flm_debias import cross_fit_logit, compute_influence_functions
from src.targets import compute_targets, targets_to_df
from src.plots import make_all_plots
from src.utils import ensure_dirs, log, save_resolved_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Digital Twins Commute-Choice Pipeline")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Seeded fake choices, no API")
    group.add_argument("--api-run", action="store_true", help="Real Anthropic API calls")
    p.add_argument("--skip-report", action="store_true", help="Skip LaTeX PDF compilation")
    p.add_argument("--n", type=int, default=None, help="Override n_per_job_type (for quick tests)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config()
    cfg = resolve_config(cfg)

    # Apply overrides
    if args.dry_run:
        cfg["experiment"]["use_api"] = False
    elif args.api_run:
        cfg["experiment"]["use_api"] = True

    if args.n is not None:
        cfg["experiment"]["n_per_job_type"] = args.n

    validate_config(cfg)

    n_total = cfg["experiment"]["n_per_job_type"] * len(cfg["randomization"]["job_types"])
    mode = "DRY-RUN" if not cfg["experiment"]["use_api"] else f"API:{cfg['anthropic']['model']}"
    log(f"=== Digital Twins Commute-Choice Experiment [{mode}] ===")
    log(f"    n_per_job_type={cfg['experiment']['n_per_job_type']}  total={n_total}")

    ensure_dirs(
        "outputs/data", "outputs/estimates", "outputs/figures",
        "outputs/tables", "outputs/report", "outputs/logs"
    )
    save_resolved_config(cfg, "outputs/logs")

    # --- Phase 1: Simulate design ---
    log("Phase 1: Simulating experimental design...")
    design = simulate_design(cfg)
    design.to_csv("outputs/data/design_pre_llm.csv", index=False)
    log(f"  {len(design)} observations generated.")

    # --- Phase 2: LLM route choices ---
    log("Phase 2: Running route-choice experiment...")
    panel = run_experiment(design, cfg, verbose=True)
    panel.to_csv("outputs/data/person_level_panel.csv", index=False)
    if cfg.get("output", {}).get("save_parquet", True):
        panel.to_parquet("outputs/data/person_level_panel.parquet", index=False)

    import jsonlines
    with jsonlines.open("outputs/data/prompts.jsonl", mode="w") as w:
        for _, row in panel.iterrows():
            w.write({"obs_id": int(row["obs_id"]), "prompt": row.get("prompt_text", "")})

    log(f"  Coastal choice rate: {panel['choose_coastal'].mean():.3f}")

    # --- Phase 3: Design matrix ---
    log("Phase 3: Building design matrix...")
    dm = build_design_matrix(panel, cfg)
    W, Y, D = dm["W"], dm["Y"], dm["D"]
    A, B = dm["A"].values, dm["B"].values
    n_A = A.shape[1]
    dm["W_df"].assign(obs_id=dm["df"]["obs_id"].values).to_csv(
        "outputs/data/estimation_design_matrix.csv", index=False
    )
    log(f"  Design matrix: {W.shape[1]} features × {len(Y)} obs.")

    # --- Phase 4: Logistic regression ---
    log("Phase 4: Estimating logit model...")
    ridge_alpha = cfg.get("estimation", {}).get("ridge_alpha", 0.01)
    fit = fit_logit(W, Y, ridge_alpha=ridge_alpha)
    log(f"  Fitted coastal share: {fit['p'].mean():.4f}  condition_number: {fit['condition_number']:.1f}")

    pd.DataFrame({"feature": dm["feature_names"], "coefficient": fit["theta"]}).to_csv(
        "outputs/estimates/logit_coefficients.csv", index=False
    )
    dm["df"][["obs_id"]].assign(p_hat=fit["p"], choose_coastal=Y).to_csv(
        "outputs/estimates/logit_fitted_values.csv", index=False
    )

    # --- Phase 5: FLM debiasing ---
    log("Phase 5: FLM debiasing (cross-fitting)...")
    cross_fit = cross_fit_logit(W, Y, cfg)
    flm = compute_influence_functions(cross_fit, W, Y, A, B, D, n_A)
    log(f"  AME (debiased) = {flm['tau_ame_debiased']:.4f}  SE = {flm['se_ame']:.4f}")

    if_df = dm["df"][["obs_id"]].copy()
    if_df["phi_ame"] = flm["phi_ame"]
    if_df["b_i"] = flm["b_i"]
    if_df["me_i"] = flm["me_i"]
    if_df.to_csv("outputs/estimates/influence_functions.csv", index=False)

    # --- Phase 6: Target parameters ---
    log("Phase 6: Computing target parameters...")
    target_result = compute_targets(panel, dm, fit, flm, cfg)
    obs_level = target_result["obs_level"]
    obs_level.to_csv("outputs/estimates/observation_level_targets.csv", index=False)
    targets_df = targets_to_df(target_result)
    targets_df.to_csv("outputs/tables/target_parameters.csv", index=False)
    log(f"  Sunny premium: {target_result['sunny_premium']:.4f}")
    log(f"  Avg D*: {target_result['targets']['avg_dstar']:.2f} min")

    # --- Phase 7: Plots ---
    log("Phase 7: Generating 18 plots...")
    make_all_plots(panel, dm, fit, target_result, "outputs/figures")

    # --- Phase 8: Report ---
    if not args.skip_report:
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location("compile_report", pathlib.Path(__file__).parent / "07_compile_report.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        log("Phase 8: Compiling report...")
        try:
            mod.main()
        except Exception as e:
            log(f"  Report compilation skipped: {e}")
    else:
        log("Phase 8: Report skipped (--skip-report).")

    log("=== Pipeline complete. ===")
    log(f"    Outputs: outputs/")


if __name__ == "__main__":
    main()

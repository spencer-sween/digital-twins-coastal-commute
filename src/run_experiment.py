"""
Phase 2: Run LLM route-choice calls for each observation.
Adds chosen_route and metadata columns to the design DataFrame.
"""
import numpy as np
import pandas as pd

from src.prompts import render_prompt
from src.api_client import get_route_choice


def run_experiment(design: pd.DataFrame, cfg: dict, verbose: bool = True) -> pd.DataFrame:
    seed = cfg["experiment"]["random_seed"]
    rng = np.random.default_rng(seed + 1)  # separate seed from design
    model = cfg["anthropic"]["model"]
    n = len(design)

    results = []
    errors = 0
    for i, row in design.iterrows():
        if verbose and (True):
            print(f"  [{i+1}/{n}] obs_id={row['obs_id']} job={row['job_type']}", end="")

        prompt_text = render_prompt(row.to_dict())
        result = get_route_choice(prompt_text, cfg, rng)

        if result["status"] in ("ok", "dry_run"):
            d = result["data"]
            record = {
                "obs_id": row["obs_id"],
                "prompt_text": prompt_text,
                "chosen_route": d["chosen_route"],
                "choose_coastal": int(d["chosen_route"] == "coastal"),
                "main_reason": d["main_reason"],
                "secondary_reason": d["secondary_reason"],
                "confidence": d["confidence"],
                "would_reconsider_if_difference_changed_by_minutes": d[
                    "would_reconsider_if_difference_changed_by_minutes"
                ],
                "one_sentence_summary": d["one_sentence_summary"],
                "api_model": model,
                "api_status": result["status"],
                "api_error": None,
                "random_seed": seed,
            }
        else:
            errors += 1
            record = {
                "obs_id": row["obs_id"],
                "prompt_text": prompt_text,
                "chosen_route": None,
                "choose_coastal": None,
                "main_reason": None,
                "secondary_reason": None,
                "confidence": None,
                "would_reconsider_if_difference_changed_by_minutes": None,
                "one_sentence_summary": None,
                "api_model": model,
                "api_status": "error",
                "api_error": result["error"],
                "random_seed": seed,
            }

        if verbose and (True):
            status = record["chosen_route"] or "ERROR"
            print(f" → {status}")

        results.append(record)

    response_df = pd.DataFrame(results)
    panel = design.merge(response_df, on="obs_id", how="left")

    if verbose:
        print(f"\n  Done. {n - errors}/{n} successful calls.")
        if errors:
            print(f"  WARNING: {errors} failed calls — api_error column has details.")

    return panel

#!/usr/bin/env python3
"""Phase 2: Run LLM route-choice calls (or dry-run with seeded fake choices)."""
import sys
import json
import jsonlines
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, resolve_config, validate_config
from src.run_experiment import run_experiment
from src.utils import ensure_dirs, log


def main() -> None:
    cfg = load_config()
    cfg = resolve_config(cfg)
    validate_config(cfg)

    import pandas as pd
    design = pd.read_csv("outputs/data/design_pre_llm.csv")

    mode = "DRY-RUN" if not cfg["experiment"]["use_api"] else f"API ({cfg['anthropic']['model']})"
    log(f"Running route-choice experiment [{mode}] — {len(design)} observations...")

    panel = run_experiment(design, cfg, verbose=True)

    ensure_dirs("outputs/data")
    panel.to_csv("outputs/data/person_level_panel.csv", index=False)

    if cfg.get("output", {}).get("save_parquet", True):
        panel.to_parquet("outputs/data/person_level_panel.parquet", index=False)

    # Save prompts and raw responses as jsonlines
    if cfg.get("output", {}).get("save_jsonl", True):
        with jsonlines.open("outputs/data/prompts.jsonl", mode="w") as writer:
            for _, row in panel.iterrows():
                writer.write({"obs_id": int(row["obs_id"]), "prompt_text": row.get("prompt_text", "")})

        with jsonlines.open("outputs/data/api_raw_responses.jsonl", mode="w") as writer:
            for _, row in panel.iterrows():
                writer.write({
                    "obs_id": int(row["obs_id"]),
                    "chosen_route": row.get("chosen_route"),
                    "confidence": row.get("confidence"),
                    "api_status": row.get("api_status"),
                    "api_error": row.get("api_error"),
                })

    n_success = panel["api_status"].isin(["ok", "dry_run"]).sum()
    n_error = panel["api_status"].eq("error").sum()
    log(f"Done. {n_success} successful, {n_error} errors.")
    log(f"Coastal choice rate: {panel['choose_coastal'].mean():.3f}")


if __name__ == "__main__":
    main()

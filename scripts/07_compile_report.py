#!/usr/bin/env python3
"""Phase 8: Compile LaTeX PDF report. Skip gracefully if pdflatex unavailable."""
import sys
import os
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config, resolve_config
from src.utils import ensure_dirs, log


def check_latex() -> bool:
    try:
        subprocess.run(["pdflatex", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main(skip_report=False) -> None:
    cfg = load_config()
    cfg = resolve_config(cfg)
    author = cfg.get("report", {}).get("author_name", "Spencer Sween")

    if skip_report:
        log("--skip-report flag set. Skipping LaTeX compilation.")
        return

    if not check_latex():
        log("WARNING: pdflatex not found. Skipping PDF compilation. Install LaTeX to enable.")
        log("The .tex source is available in reports/")
        return

    ensure_dirs("outputs/report")
    report_src = Path("reports/report.tex")
    if not report_src.exists():
        log("reports/report.tex not found. Run after report.tex is generated.")
        return

    log("Compiling LaTeX report...")
    for _ in range(2):  # two passes for references
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", "outputs/report", str(report_src)],
            capture_output=True,
        )
    if result.returncode == 0:
        log("PDF report compiled: outputs/report/commute_digital_twins_report.pdf")
    else:
        log("LaTeX compilation failed. Check outputs/report/*.log for details.")


if __name__ == "__main__":
    skip = "--skip-report" in sys.argv
    main(skip_report=skip)

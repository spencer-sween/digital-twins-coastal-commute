# Experiment Summary: Digital Twins Commute-Choice Experiment

> Derived from `INSTRUCTIONS.md`. Optimized for use with **Claude** (Anthropic) and **Claude Code** (CLI).
> Note: Several references in the original spec mention OpenAI/GPT — these are flagged below and should be replaced with Claude equivalents before implementation.

---

## Goal

Build a fully reproducible, end-to-end **digital twins discrete-choice experiment** that:

1. Simulates Wednesday evening commuters choosing between a freeway route and a coastal route home from the Intuit campus in Torrey Highlands, San Diego to Encinitas.
2. Uses an LLM to generate route choices under randomized, controlled pre-decision attributes (job type, release time, fatigue, weather, travel times).
3. Estimates a **flexible logit demand model** where both the intercept and the time-differential slope vary by observed characteristics.
4. Applies an **FLM-style debiasing layer** to produce influence-function-based inference for target econometric functionals.
5. Produces a full suite of output files, plots, and a compiled LaTeX PDF report (with a LinkedIn/blog-style writeup).

**This is a demonstration of digital twin methodology and prompt-driven discrete-choice design, not a claim about real human commuting behavior.**

---

## Phases / Main Steps

### Phase 0 — Configuration and Validation
- User fills in `configs/experiment_config.yaml` with route distances, projected travel times, simulation counts, and API settings.
- `scripts/00_validate_config.py` must fail loudly if required fields are null.

### Phase 1 — Experimental Design Generation (`01_simulate_design.py`)
- Draw person-level observations across 5 job types x 4 release times x weather states x fatigue states.
- Apply bounded heavy-tailed jitter (Student-t, df=3, scale=0.32) to projected route times, respecting physical speed bounds (5-80 mph).
- Export base and jittered travel times.

### Phase 2 — Prompt Generation and LLM Route-Choice Calls (`02_run_llm_choices.py`)
- Render a fixed prompt template for each observation (job type, release time, fatigue description, weather description, route times).
- Call the LLM API with structured output (JSON schema enforced).
- Support `USE_API=false` (dry-run with seeded fake choices) and `USE_API=true` (real API calls).
- Record all API errors per observation; do not silently drop observations.

### Phase 3 — Design Matrix Construction (`03_build_design_matrix.py`)
- Construct the estimation-ready matrix with flexible intercept basis `A(X_i)` and slope basis `B(X_i)` interacted with the coastal-minus-freeway time differential `D_i`.
- Basis includes: intercept, job type dummies, release time dummies, fatigue dummies, weather dummy, and configurable interaction terms.

### Phase 4 — Logit Estimation (`04_estimate_logit.py`)
- Estimate `P(Y_i=1 | X_i, D_i) = L(a(X_i) + b(X_i) * D_i)` where L is the logistic function.
- Support unregularized and ridge-regularized logistic regression (cross-validated ridge as option).
- Export: coefficients, fitted probabilities, score contributions, Hessian (matrix and long format), eigenvalues, condition numbers.

### Phase 5 — FLM-Style Debiasing (`05_flm_debias.py`)
- Compute influence functions for smooth target functionals using the sample analog of the semiparametric influence function.
- Implement K-fold cross-fitting (default K=5).
- Support three Hessian inversion strategies: direct, ridge-regularized, spectral with eigenvalue flooring.
- Log and export every regularization choice; never regularize silently.

### Phase 6 — Target Parameter Estimation (`targets.py`)
Compute and export debiased estimates and standard errors for:
- Average coastal choice probability and observed choice rate (overall and by job type).
- Conditional slope `b(X_i)` and intercept `a(X_i)` distributions.
- Average marginal effect (AME) of the time differential.
- Average semi-elasticity and elasticity.
- Route-time willingness-to-take threshold `D_i* = -a(X_i)/b(X_i)` (with trimming for near-zero slopes).
- Sunny-day coastal premium.
- Fatigue-state and release-time contrasts.

### Phase 7 — Plots (`06_make_plots.py`)
18 required plots (PNG and PDF), including choice shares by subgroup, predicted probability curves, distributions of conditional parameters, and job-specific marginal effects with confidence intervals.

### Phase 8 — Report Compilation (`07_compile_report.py`)
- Compile a LaTeX PDF with 12 sections (title page through LinkedIn/blog writeup).
- Include a polished LinkedIn/blog post stressing honest limitations and methodological value.

### Phase 9 — GitHub Push (conditional on user authorization)
- Clone or use local path, optionally create branch, run validation + dry-run, commit, push.

---

## Key Personas / Roles

| Role | Description |
|---|---|
| **Coding agent** | Claude Code (or similar) that reads this spec, asks the user for required inputs, then builds the full pipeline. Must not assume defaults without explicit user approval. |
| **Simulated commuter (digital twin)** | An LLM instance prompted to act as a human commuter making a realistic route-choice decision on a Wednesday evening. |
| **User / researcher** | Provides route distances, projected travel times, simulation counts, API model, GitHub repo, and report authorship preferences before code is written. |

---

## Technical Requirements

### Language and Stack
- **Python** (primary implementation language)
- Standard scientific Python stack: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openpyxl`, `pyyaml`, `jsonlines`
- LaTeX for PDF report compilation
- `pytest` for tests

### LLM / API
- **Structured output** JSON schema enforcement required for route-choice calls.
- API key must come from environment variable (see OpenAI references section below for Claude equivalent).
- Must support dry-run (no API calls) and API-run modes.
- Retry logic: max 4 retries, 60-second timeout.

### Reproducibility
- Fixed seeds throughout; resolved config written to `outputs/logs/config_resolved.yaml`.
- Outputs folder excluded from git by default (`.gitignore`).

### Project Structure
```
commute-digital-twins/
  configs/experiment_config.yaml
  src/          # Core modules (config, simulate_design, prompts, api_client, ...)
  scripts/      # Numbered pipeline scripts (00-07) + run_all.py
  tests/        # pytest test suite (11 required tests)
  reports/      # LaTeX source
  outputs/      # Generated by pipeline; not committed by default
```

### Key Output Files
- `outputs/data/person_level_panel.csv` and `.parquet`
- `outputs/estimates/` — coefficients, Hessian, scores, influence functions, target parameters
- `outputs/tables/` — summary stats, choice shares, subgroup estimates
- `outputs/figures/` — 18+ plots in PNG and PDF
- `outputs/report/commute_digital_twins_report.pdf`
- `outputs/report/linkedin_post.md`
- `outputs/commute_digital_twins_outputs.xlsx` (10-tab workbook)

---

## OpenAI / ChatGPT References Needing Replacement with Claude Equivalents

The original `INSTRUCTIONS.md` contains several OpenAI-specific references that must be updated before building with Claude:

| Location in spec | Original reference | Claude equivalent |
|---|---|---|
| Required user inputs (Q2) | "What OpenAI model should be used for the route-choice calls?" | Ask what **Claude model** to use (e.g., `claude-haiku-3-5` for cost-efficiency, `claude-sonnet-4-5` for quality). |
| `experiment_config.yaml` section `openai:` | `model: null`, `api_key_env_var: OPENAI_API_KEY` | Replace with `anthropic:` block; use `ANTHROPIC_API_KEY` env var. |
| API client implementation | Calls to OpenAI structured outputs API | Use Anthropic Python SDK (`anthropic` package); implement structured output via Claude tool-use with a defined input schema. |
| `.env.example` | `OPENAI_API_KEY=...` | Replace with `ANTHROPIC_API_KEY=...` |
| Structured output schema enforcement | "Use OpenAI structured outputs or the equivalent" | Use Claude tool-use with a strict JSON input schema, or prompt Claude to return valid JSON and validate with a schema parser. |
| Acceptance criterion #4 | "after setting `OPENAI_API_KEY`" | "after setting `ANTHROPIC_API_KEY`" |

**Action required before writing code:** Confirm which Claude model to use. Update all config templates, `.env.example`, and `src/api_client.py` to use the `anthropic` SDK. Enforce structured JSON output via Claude tool-use schemas rather than OpenAI's native structured output mode.

---

## Open Questions and Gaps

1. **Claude model selection**: The spec assumes OpenAI throughout. The user must specify a Claude model before code is written. Recommended starting points:
   - `claude-haiku-3-5` for large simulation runs (low cost, fast)
   - `claude-sonnet-4-5` or `claude-sonnet-4-6` for higher fidelity responses

2. **Structured output enforcement with Claude**: OpenAI has a native "structured outputs" mode. With Claude, structured JSON is enforced via tool-use schemas or strong prompting plus validation. The `api_client.py` must implement this pattern explicitly.

3. **Required user inputs not yet provided** (per spec, must ask before writing any code):
   - GitHub repo URL or local path, plus branch preferences and push authorization
   - Route distances (`FREEWAY_DISTANCE_MILES`, `COASTAL_DISTANCE_MILES`)
   - Projected route times for each of 8 route x release-time combinations
   - Number of simulations per job type (pilot: 100, demo: 600, precise: 1,000+)
   - Whether to start with dry-run or real API calls
   - Whether to include the user's name in the LinkedIn/blog writeup

4. **LaTeX compilation environment**: The report step requires `pdflatex` or equivalent. The pipeline must check for LaTeX availability or make the report step gracefully skippable (`--skip-report` flag is already specified).

5. **Batching**: The spec notes that batch API execution should be designed in but synchronous is acceptable for the first build. Claude's Message Batches API supports async batching and can be added in a later iteration.

6. **Cross-fitting and future ML nuisances**: The FLM debiasing layer is built to accept richer ML nuisances later. The first implementation uses finite-dimensional logit only; keep module boundaries clean to enable future swaps.

7. **Outputs git exclusion**: The spec excludes `outputs/` from git by default. Confirm this preference with the user before the GitHub push step, since outputs include model responses and prompts.

---

## Quick Reference: Run Commands

```bash
# Validate config (must run first)
python scripts/00_validate_config.py

# Full pipeline, dry-run (no API calls, fake choices)
python scripts/run_all.py --dry-run

# Full pipeline, real API calls
python scripts/run_all.py --api-run

# Skip LaTeX report compilation
python scripts/run_all.py --api-run --skip-report

# Run tests
pytest
```

### Individual pipeline steps
```bash
python scripts/01_simulate_design.py
python scripts/02_run_llm_choices.py
python scripts/03_build_design_matrix.py
python scripts/04_estimate_logit.py
python scripts/05_flm_debias.py
python scripts/06_make_plots.py
python scripts/07_compile_report.py
```

# Digital Twins Commute-Choice Experiment

A fully reproducible Python pipeline that simulates Wednesday evening commuters choosing between a freeway and a coastal route home from the Intuit campus in Torrey Highlands, San Diego to Encinitas, CA. Each "commuter" is a Claude LLM agent — a digital twin — given a randomized persona and asked to make a realistic route-choice decision.

The pipeline estimates a flexible logit demand model and applies FLM-style debiased inference to recover marginal effects, elasticities, and willingness-to-take thresholds with valid standard errors.

---

## What This Is

I got interested in digital twins and wanted to see what you could learn from running a controlled discrete-choice experiment entirely inside an LLM. The idea: if you randomize the treatment (travel time differential) across thousands of synthetic agents with varied observable characteristics, you can recover a demand curve — not for real humans, but for the LLM's internal model of a human commuter.

The routes are real. The campus is real. The drive times are calibrated to actual Wednesday evening traffic. The commuters are not.

**Full technical specification**: `markdown/INSTRUCTIONS_CLEAN.md`  
**Econometric theory**: `markdown/ECONOMETRIC_THEORY.md`

---

## The Experiment

Each simulated observation is one Wednesday evening commute decision. The agent is told:

- They work at the **Intuit campus in Torrey Highlands**, San Diego
- They commute home to **Encinitas**
- They drive a **2019 Toyota Tacoma**, gas tank about half full
- Their last meeting just ended

They choose between:

| Route | Description | Distance |
|-------|-------------|----------|
| **A — Freeway** | CA-56 → I-5 North → Cardiff exit | 14 miles |
| **B — Coastal** | West to Coast Hwy 101 → north through Del Mar, Solana Beach, Cardiff | 16 miles |

Each agent observes their randomized:
- **Job type**: data science, finance, marketing, product design, software engineering
- **Release time**: 5:00, 5:15, 5:30, or 5:45 PM
- **Fatigue state**: refreshed, tired, mentally worn down, overwhelmed (correlated with release time)
- **Weather**: sunny or cloudy (50/50, independent)
- **Travel times**: base projected times plus bounded heavy-tailed jitter (Student-*t*, df=3, logit-space)

---

## The Model

The econometric object is the route-choice response to the **coastal-minus-freeway time differential**:

$$D_i = t_i^{\text{coastal}} - t_i^{\text{freeway}}$$

$$P(Y_i = 1 \mid X_i, D_i) = \Lambda\!\left(a(X_i) + b(X_i) \cdot D_i\right)$$

where $\Lambda$ is the logistic CDF, $a(X_i)$ is a flexible intercept, and $b(X_i) < 0$ is the time-sensitivity slope. Both vary by job type, fatigue, weather, release time, and their interactions.

**Target parameters** (all with FLM-debiased standard errors via K=5 cross-fitting):

| Symbol | Meaning |
|--------|---------|
| $\tau_p = E[p_i]$ | Average coastal choice probability |
| $b(X_i)$ | Conditional time-sensitivity slope |
| $\text{AME} = E[p_i(1-p_i)b(X_i)]$ | Average marginal effect of $D_i$ |
| $D_i^* = -a(X_i)/b(X_i)$ | Willingness-to-take threshold (minutes of coastal premium at indifference) |
| Sunny premium | $E[p_i \mid \text{sunny}] - E[p_i \mid \text{cloudy}]$ |
| Fatigue contrasts | vs. refreshed baseline |

See `markdown/ECONOMETRIC_THEORY.md` for full derivations.

---

## Setup

```bash
git clone https://github.com/spencer-sween/digital-twins-coastal-commute.git
cd digital-twins-coastal-commute
pip install -r requirements.txt

cp .env.example .env
# Add your Anthropic API key to .env:
# ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the Pipeline

```bash
# Validate config (run this first)
python scripts/00_validate_config.py

# Dry-run — seeded fake choices, no API calls, validates end-to-end
python scripts/run_all.py --dry-run

# Full API run — 5000 simulations, ~$6.50 at Haiku 4.5 pricing
python scripts/run_all.py --api-run

# Skip LaTeX report if pdflatex isn't installed
python scripts/run_all.py --api-run --skip-report

# Quick test with fewer observations
python scripts/run_all.py --api-run --n 10
```

Progress prints every call for the first 5, then every 100. Full run takes roughly 20–40 minutes depending on API rate limits.

---

## Pipeline Stages

| Script | Phase | What it does |
|--------|-------|--------------|
| `00_validate_config.py` | — | Fail loudly if config has null required fields |
| `01_simulate_design.py` | 1 | Generate 5000 observations with jittered travel times |
| `02_run_llm_choices.py` | 2 | Call Claude API for each observation; record route choice + reasoning |
| `03_build_design_matrix.py` | 3 | Build flexible logit design matrix $W_i = [A(X_i);\ D_i B(X_i)]$ |
| `04_estimate_logit.py` | 4 | Ridge logit; export coefficients, Hessian, scores, eigenvalues |
| `05_flm_debias.py` | 5 | K-fold cross-fitting; compute influence functions; debiased AME + SE |
| `06_make_plots.py` | 6 | Generate all 18 plots (PNG + PDF) |
| `07_compile_report.py` | 7 | Compile LaTeX PDF report (requires pdflatex) |

---

## Outputs

```
outputs/
  data/
    person_level_panel.csv        ← one row per agent; all covariates + LLM response
    person_level_panel.parquet
    estimation_design_matrix.csv  ← W matrix fed to logit
    prompts.jsonl                 ← full prompt text for each agent
  estimates/
    logit_coefficients.csv        ← theta_hat
    logit_fitted_values.csv       ← p_hat per agent
    hessian_matrix.csv            ← full k×k Hessian
    hessian_long.csv              ← long-format Hessian for inspection
    hessian_eigenvalues.csv       ← condition number diagnostics
    influence_functions.csv       ← phi_i per agent
    observation_level_targets.csv ← a_i, b_i, ME_i, D*_i per agent
  figures/
    01_choice_share_by_job.{png,pdf}
    02_choice_share_by_release_time.{png,pdf}
    03_choice_share_by_fatigue.{png,pdf}
    04_choice_share_by_weather.{png,pdf}
    05–07_dist_*.{png,pdf}        ← travel time distributions
    08–09_predicted_prob_*.{png,pdf}
    10–14_dist_*.{png,pdf}        ← b(X), a(X), ME, elasticity, D* distributions
    15–16_job_*.{png,pdf}         ← job-specific AME and elasticity
    17_sunny_vs_cloudy.{png,pdf}
    18_fatigue_contrasts.{png,pdf}
  tables/
    target_parameters.csv         ← all debiased target estimates
  logs/
    config_resolved.yaml          ← exact config used for this run
```

---

## Configuration

All parameters live in `configs/experiment_config.yaml`. Key settings:

```yaml
experiment:
  n_per_job_type: 1000    # × 5 job types = 5000 total simulations
  use_api: false          # set true for real API calls

anthropic:
  model: claude-haiku-4-5

routes:
  freeway_distance_miles: 14.0
  coastal_distance_miles: 16.0
  base_times_freeway:     # Wednesday evening projections (minutes)
    "17:00": 31
    "17:15": 31
    "17:30": 31
    "17:45": 29
  base_times_coastal:
    "17:00": 42
    "17:15": 42
    "17:30": 41
    "17:45": 40

flm:
  n_folds: 5
  hessian_inversion: ridge   # direct | ridge | spectral
```

---

## Tests

```bash
pytest
```

18 tests covering: jitter bounds, design reproducibility, prompt schema validation, design matrix construction, and target parameter computation.

---

## Cost Estimate

At `claude-haiku-4-5` pricing ($1.00/M input · $5.00/M output):

| Run | Calls | Est. Cost |
|-----|-------|-----------|
| Dry-run | 0 | $0.00 |
| Quick test (`--n 10`) | 50 | ~$0.03 |
| Full run (`--n 1000`) | 5,000 | ~$6.50 |

---

## Limitations

This is a demonstration of digital twin methodology, not a study of real human commuting behavior. Inference is about the LLM's decision rule given the prompt design — not about the population of Intuit employees. The time differential $D_i$ is randomized by design, so no natural-experiment identification is claimed or needed. See `markdown/ECONOMETRIC_THEORY.md` §7–8 for a full discussion.

---

## Stack

- **LLM**: Anthropic `claude-haiku-4-5` via the `anthropic` Python SDK
- **Estimation**: `scikit-learn` logistic regression, `numpy`/`scipy` for Hessian and influence functions
- **Data**: `pandas`, `pyarrow`
- **Plots**: `matplotlib`
- **Report**: LaTeX (optional)

---

*Built by Spencer Sween. Got curious about digital twins, decided to have a little fun.*

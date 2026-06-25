*I am SO over taking ugly sludge through freeway traffic every Wednesday evening. So I asked: does AI have a soul — will it, like me, sometimes just take the scenic coastal route home, even when the freeway is faster? Let's find out with Digital Twins for Intuit employees who live in Encinitas.*

> **HEADLINE RESULT: Guess what — data scientists DO like to have fun. 24.1% coastal-takers, highest of any job type (tied with product design, but with a larger elasticity).**

---

# Digital Twins Commute-Choice Experiment

A fully reproducible Python pipeline that simulates Wednesday evening commuters choosing between a freeway and a coastal route home from the Intuit campus in Torrey Highlands, San Diego to Encinitas, CA. Each "commuter" is a Claude LLM agent — a digital twin — given a randomized persona and asked to make a realistic route-choice decision.

The pipeline estimates a flexible logit demand model and applies FLM-style debiased inference to recover marginal effects, elasticities, and willingness-to-take thresholds with valid standard errors.

---

## What This Is

I got interested in digital twins and wanted to see what you could learn from running a controlled discrete-choice experiment entirely inside an LLM. The idea: if you randomize the treatment (travel time differential) across thousands of synthetic agents with varied observable characteristics, you can recover a demand curve — not for real humans, but for the LLM's internal model of a human commuter.

The routes are real. The campus is real. The drive times are calibrated to actual Wednesday evening traffic. The commuters are not.

**Full technical specification**: `markdown/INSTRUCTIONS_CLEAN.md`  
**Econometric theory and derivations**: `markdown/ECONOMETRIC_THEORY.md`

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
| **B — Coastal** | West to Coast Hwy 101 → north through Del Mar, Solana Beach, Cardiff-by-the-Sea | 16 miles |

Each agent observes their randomized:
- **Job type**: data science, finance, marketing, product design, software engineering
- **Release time**: 5:00, 5:15, 5:30, or 5:45 PM
- **Fatigue state**: refreshed, tired, mentally worn down, overwhelmed (correlated with release time)
- **Weather**: sunny or cloudy (50/50, independent)
- **Travel times**: base projected times plus bounded heavy-tailed jitter (Student-*t*, df=3, logit-space)

---

## Prompt Template

Every agent receives a prompt of this form. Template variables are filled at runtime from the randomized design:

```
You are a human commuter participating in a route-choice study.

You work at the Intuit campus in Torrey Highlands, San Diego. You commute home
on Wednesdays to Encinitas, a beach town in North County San Diego. You drive a
2019 Toyota Tacoma. Your gas tank is about half full.

Today is Wednesday. Your last meeting has just ended, and you are deciding
which route to take home.

Your job type is: {job_type}

Your last meeting ended at: {release_time_label}

Your current fatigue state is: {fatigue_state}

Fatigue description:
{fatigue_description}

Weather and visibility:
{weather_description}

You have two available routes.

Route A: Standard freeway route
- Description: Drive from the Torrey Highlands campus to CA-56, connect to I-5
  North, exit near Cardiff, and continue toward Encinitas.
- Estimated travel time today: {freeway_time_minutes:.1f} minutes.
- Practical features: mostly freeway driving, fewer traffic lights, more direct
  routing, less scenery.

Route B: Coastal route
- Description: Drive west from the Torrey Highlands area toward the coast,
  connect toward Coast Highway 101, then continue north through Del Mar,
  Solana Beach, and Cardiff-by-the-Sea toward Encinitas.
- Estimated travel time today: {coastal_time_minutes:.1f} minutes.
- Practical features: more local street driving, streetlights and stop signs,
  possible pedestrian and beach traffic, more coastal scenery.

Choose the route you would actually take today.

Decision instructions:
- Answer as a normal human commuter, not as an optimization algorithm.
- Consider travel time, uncertainty, fatigue, weather, scenery, traffic lights,
  stop signs, and the kind of driving involved.
- Do not assume that the fastest route is always preferred.
- Do not make exaggerated assumptions based only on job type.

Return valid JSON only.
```

The model is instructed via system prompt to return only valid JSON matching a strict 6-field schema (`chosen_route`, `main_reason`, `secondary_reason`, `confidence`, `would_reconsider_if_difference_changed_by_minutes`, `one_sentence_summary`). Responses are validated with `jsonschema` before storing.

---

## The Model

**Random utility setup.** Agent $i$ computes latent utility differences between routes. Let $U_i^c$ and $U_i^f$ be the utilities of choosing coastal vs. freeway. Under a random utility model (McFadden 1974):

$$U_i^c - U_i^f = a(X_i) + b(X_i) \cdot D_i + \varepsilon_i$$

where $\varepsilon_i$ is a logistic error (i.i.d. across agents). The agent chooses coastal iff $U_i^c > U_i^f$. Integrating out $\varepsilon_i$ gives the logit probability:

$$P(Y_i = 1 \mid X_i, D_i) = \Lambda\left(a(X_i) + b(X_i) \cdot D_i\right), \qquad \Lambda(v) = \frac{1}{1+e^{-v}}$$

**Treatment variable.** The treatment is the coastal-minus-freeway time differential in minutes (not logged):

$$D_i = t_i^{\text{coastal}} - t_i^{\text{freeway}}$$

$D_i > 0$ means coastal is slower. Because $D_i$ is drawn from a bounded jitter distribution independently of $X_i$, it is mean-independent of agent characteristics by design.

**Flexible specification.** Both the intercept $a(X_i)$ and the slope $b(X_i)$ are approximated by linear basis expansions over job type, release time, fatigue state, weather, and their pairwise interactions. This lets the time sensitivity differ by job type and fatigue rather than forcing a single slope. We expect $b(X_i) < 0$ everywhere.

**Estimation and inference.** $\theta$ is estimated by maximum likelihood (ridge-regularized logit). Target parameters — average marginal effects, elasticities, willingness-to-take thresholds — are debiased via FLM-style influence functions with K=5 cross-fitting. Job-type confidence intervals use Bonferroni correction (z = 2.576, α = 0.05 across 5 groups).

See `markdown/ECONOMETRIC_THEORY.md` for full derivations including the score, Hessian, and cross-fitting protocol.

---

## Results  *(n = 500, 100 per job type)*

### Panel A — Coastal Route Choice Share by Job Type

| Job Type | P(coastal) | SE | 95% CI (Bonferroni) | n |
|----------|:----------:|:---:|:-------------------:|:-:|
| Data Science       | 0.241 | 0.033 | [0.155, 0.326] | 100 |
| Finance            | 0.175 | 0.029 | [0.102, 0.249] | 100 |
| Marketing          | 0.188 | 0.028 | [0.115, 0.261] | 100 |
| Product Design     | 0.241 | 0.034 | [0.153, 0.329] | 100 |
| Software Eng.      | 0.194 | 0.030 | [0.118, 0.271] | 100 |
| **Overall**        | **0.202** | — | — | **500** |

### Panel B — Average Elasticity of Coastal Choice w.r.t. Time Differential (*EL*)

Elasticity *EL_i = D_i (1 − p_i) b(X_i)* is the point elasticity of P(coastal) with respect to the time differential: a 1% increase in D_i multiplies P(coastal) by approximately (1 + EL_i/100). Concretely, EL = −5.7 means a 10% larger time premium reduces P(coastal) by about 57% relative to its baseline — e.g., from 0.20 to 0.086.

Note: D_i enters the model **in levels** (minutes), not logged. EL_i is a derived quantity from the level-level logit — it is the standard arc elasticity (∂p/∂D)·(D/p) evaluated at each observation, not a regression coefficient from a log-log model.

| Job Type | Avg *EL* | SE | 95% CI (Bonferroni) | n |
|----------|:--------:|:---:|:-------------------:|:-:|
| Data Science       | −6.406 | 0.979 | [−8.928, −3.884] | 100 |
| Finance            | −5.977 | 0.743 | [−7.890, −4.063] | 100 |
| Marketing          | −5.107 | 0.767 | [−7.083, −3.131] | 100 |
| Product Design     | −5.694 | 0.820 | [−7.806, −3.581] | 100 |
| Software Eng.      | −5.300 | 0.840 | [−7.465, −3.134] | 100 |
| **Overall**        | **−5.697** | 0.002 | — | **500** |

**Why do data scientists have both the highest coastal take rate AND the largest elasticity magnitude?** These are driven by separate parameters:

- **Higher take rate** (24.1%): Data scientists have the least-negative intercept $a_i = -0.36$ — the strongest unconditional preference for the coastal route. This is a structural feature of the LLM's representation of data scientists (maybe they value the decompression?).

- **Largest |EL|**: Despite this, their slope $b_i = -0.389$ is actually the *least* negative of all job types — they are structurally the *least* time-sensitive. The large elasticity comes from having the largest mean D_i in this pilot (14.0 min vs. ~9–13 for other groups), which is a random-sample artifact: EL_i = D_i·(1−p_i)·b_i scales with D_i, and with only 100 obs per group the realized mean D can vary substantially. In a larger run this ordering would likely change.

In short: the intercept tells you who likes the coast; the slope tells you who cares about time; the elasticity conflates both with realized sample D — so read the three panels together.

### Panel C — Average Willingness-to-Take Threshold (*D\**)

*D\*_i = −a(X_i) / b(X_i)* is the coastal time premium (in minutes) at which the model predicts indifference (P = 0.5). Negative values mean the agent prefers the freeway even when the coastal route is faster — i.e., the coastal scenery/driving style isn't worth even a small time cost.

| Job Type | Avg *D\** (min) | SE | 95% CI (Bonferroni) | n |
|----------|:--------------:|:---:|:-------------------:|:-:|
| Data Science       | −2.52 | 1.61 | [−6.67, 1.62] | 100 |
| Finance            | −3.13 | 0.84 | [−5.30, −0.96] | 100 |
| Marketing          | −3.82 | 1.13 | [−6.73, −0.92] | 100 |
| Product Design     | −1.35 | 1.13 | [−4.26,  1.55] | 100 |
| Software Eng.      | −1.99 | 0.76 | [−3.94, −0.04] | 100 |
| **Overall**        | **−2.56** | — | — | **500** |

> **Note:** All standard errors are influence-function based. Job-type 95% CIs use Bonferroni correction (z = 2.576) for simultaneous inference across 5 groups. Results are from a pilot run of 100 observations per job type. Full 1,000-per-job run is pending.

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

# Pilot run — 500 simulations, ~$0.65
python scripts/run_all.py --api-run --n 100

# Skip LaTeX report if pdflatex isn't installed
python scripts/run_all.py --api-run --skip-report
```

---

## Pipeline Stages

| Script | Phase | What it does |
|--------|-------|--------------|
| `00_validate_config.py` | — | Fail loudly if config has null required fields |
| `01_simulate_design.py` | 1 | Generate observations with bounded heavy-tailed jitter on travel times |
| `02_run_llm_choices.py` | 2 | Call Claude API for each agent; record route choice + reasoning |
| `03_build_design_matrix.py` | 3 | Build flexible logit design matrix $W_i = [A(X_i);\ D_i B(X_i)]$ |
| `04_estimate_logit.py` | 4 | Ridge logit; export coefficients, Hessian, scores, eigenvalues |
| `05_flm_debias.py` | 5 | K=5 cross-fitting; influence functions; debiased AME + SE |
| `06_make_plots.py` | 6 | 20 plots (PNG + PDF) including per-job distributions |
| `07_compile_report.py` | 7 | LaTeX PDF (requires pdflatex) |

---

## Outputs

```
outputs/
  data/
    person_level_panel.csv          ← one row per agent; all covariates + LLM response
    person_level_panel.parquet
    estimation_design_matrix.csv    ← W matrix fed to logit
    prompts.jsonl                   ← full prompt text for each observation
  estimates/
    logit_coefficients.csv
    logit_fitted_values.csv
    hessian_matrix.csv
    hessian_eigenvalues.csv
    influence_functions.csv
    observation_level_targets.csv   ← a_i, b_i, ME_i, EL_i, D*_i per agent
  figures/                          ← 20 plots × {PNG, PDF}
  tables/
    target_parameters.csv           ← all debiased estimates with SEs
  logs/
    config_resolved.yaml
```

---

## Configuration

All parameters live in `configs/experiment_config.yaml`. Key settings:

```yaml
experiment:
  n_per_job_type: 1000    # × 5 job types = 5000 total
  use_api: false          # set true for real API calls

anthropic:
  model: claude-haiku-4-5

routes:
  freeway_distance_miles: 14.0
  coastal_distance_miles: 16.0
  base_times_freeway:     # Wednesday evening (minutes)
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
  hessian_inversion: ridge
```

---

## Tests

```bash
pytest            # 18 tests — jitter bounds, schema validation, matrix construction, targets
```

---

## Cost Estimate

At `claude-haiku-4-5` pricing ($1.00/M input · $5.00/M output):

| Run | Calls | Est. Cost |
|-----|-------|-----------|
| Dry-run | 0 | $0.00 |
| Pilot (`--n 100`) | 500 | ~$0.65 |
| Full (`--n 1000`) | 5,000 | ~$6.50 |

---

## Limitations

This is a demonstration of digital twin methodology, not a study of real human commuting behavior. Inference is about the LLM's decision rule given the prompt design — not about real Intuit employees or the broader population. The time differential D_i is randomized by design, so no natural-experiment identification is claimed or needed. See `markdown/ECONOMETRIC_THEORY.md` §7–8 for a full discussion.

---

## Stack

- **LLM**: Anthropic `claude-haiku-4-5` via the `anthropic` Python SDK
- **Estimation**: `scikit-learn` logistic regression, `numpy`/`scipy` for Hessian and influence functions
- **Data**: `pandas`, `pyarrow`
- **Plots**: `matplotlib` with KDE overlays
- **Report**: LaTeX (optional)

---

*Built by Spencer Sween. Got curious about digital twins, decided to have a little fun.*

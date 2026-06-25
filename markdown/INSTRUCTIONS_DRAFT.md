# Agent specification: Digital twins commute-choice experiment

## Purpose

Build an end-to-end project for a digital twins commute-choice experiment. The experiment simulates Wednesday evening commuters who work at the Intuit campus in Torrey Highlands, San Diego, live in Encinitas, and choose between two routes home:

1. Standard freeway route: CA-56 to I-5 North, then toward Cardiff and Encinitas.
2. Coastal route: toward Coast Highway 101, then north through Del Mar, Solana Beach, and Cardiff-by-the-Sea.

The project must generate a person-level experimental dataset, call an LLM to obtain route choices under controlled pre-decision attributes, estimate a flexible logit demand model, construct FLM-style debiased estimates and influence functions for route-time sensitivity parameters, and produce output files, plots, and a compiled PDF report.

This markdown file is intended to be fed to a coding agent such as Codex, Claude Code, or Cursor. The first task of the agent is to digest the full specification and ask the user for the required inputs before writing production code.

---

## First action required from the coding agent

Before writing code, ask the user for the following inputs. Do not assume these values unless the user explicitly asks you to use defaults.

### Required user-provided inputs

Ask the user:

1. What GitHub repo should I write this project to?
   - Ask for either a GitHub URL or a local path to a cloned repo.
   - Ask whether to create a new branch.
   - Ask whether the agent should commit and push after building the project.

2. What OpenAI model should be used for the route-choice calls?
   - Suggested default: a low-cost structured-output capable model.
   - The script must allow this to be set via environment variable.

3. What are the route distances?
   - `FREEWAY_DISTANCE_MILES`
   - `COASTAL_DISTANCE_MILES`

4. What are the projected route times at each release time?
   - Freeway route projected minutes at:
     - 5:00 PM
     - 5:15 PM
     - 5:30 PM
     - 5:45 PM
   - Coastal route projected minutes at:
     - 5:00 PM
     - 5:15 PM
     - 5:30 PM
     - 5:45 PM

5. How many simulations should be run per job type?
   - Suggested pilot: 100 per job type, 500 total.
   - Suggested demo-quality run: 600 per job type, 3,000 total.
   - Suggested precise run: 1,000 or more per job type, 5,000 or more total.

6. Should the default randomization design be used?
   - Job types: data science, finance, marketing, product design, software engineering.
   - Release time randomized uniformly across 5:00, 5:15, 5:30, and 5:45 PM, conditional on job type.
   - Weather randomized independently as sunny or cloudy with 50 percent probability.
   - Fatigue correlated with release time using the table below.

7. Should the default bounded heavy-tailed travel-time jitter be used?
   - Recommended default:
     - Student-t degrees of freedom: 3
     - Jitter scale: 0.32
     - Lower support: no faster than 80 mph over the full route.
     - Upper support: no slower than 5 mph over the full route.

8. Does the user want the project to run API calls immediately, or only generate the prompts and dry-run simulation first?
   - The code must support both `USE_API = false` and `USE_API = true`.

9. Ask whether the final PDF should use the user's name in the LinkedIn-style writeup.
   - If yes, ask for the exact name and title to use.
   - If no, write generically in first person.

Do not ask for information that is already answered by the user. Ask only for missing constants and repo logistics.

---

## Project objective

Create a fully reproducible project with these stages:

1. Generate experimental person-level observations.
2. Generate prompt text for each observation.
3. Call the OpenAI API using structured outputs.
4. Store route choices and explanations.
5. Build estimation-ready design matrices.
6. Estimate flexible logit choice models.
7. Construct marginal effects, elasticities, and route-time willingness-to-take measures.
8. Implement an FLM-style debiasing layer for target functionals.
9. Export coefficient tables, target parameter tables, Hessian entries, score contributions, and influence-function contributions.
10. Produce plots.
11. Compile a LaTeX PDF report.
12. Include a polished, human-sounding economist-style LinkedIn/blog post about the experiment.
13. Commit and push to the GitHub repo provided by the user, if authorized.

---

## Conceptual design

Each simulated observation is one Wednesday evening commute decision.

The agent is told:

- They work at the Intuit campus in Torrey Highlands, San Diego.
- They commute home to Encinitas.
- They drive a 2019 Toyota Tacoma.
- Their gas tank is about half full.
- Their last meeting just ended.
- They have two route options:
  - Standard freeway route.
  - Coastal route through Del Mar, Solana Beach, and Cardiff-by-the-Sea.
- They observe route-specific estimated travel times before choosing.
- They observe their fatigue state and weather state.
- They are asked to choose as a normal human commuter, not as an optimization algorithm.

The econometric object is the route-choice response to the time differential:

\[
D_i = T_{i, coastal} - T_{i, freeway}
\]

The binary outcome is:

\[
Y_i = 1[\text{agent chooses coastal route}]
\]

The main model is:

\[
P(Y_i = 1 \mid X_i, D_i) =
\Lambda(a(X_i) + b(X_i)D_i)
\]

where:

- \(X_i\) is a vector of observed pre-decision characteristics.
- \(D_i\) is the coastal-minus-freeway travel-time differential.
- \(a(X_i)\) is a flexible intercept function.
- \(b(X_i)\) is a flexible slope function.
- \(\Lambda(\cdot)\) is the logistic function.

The key economic interpretation is that \(b(X_i)\) measures conditional sensitivity to the coastal route's time premium. Since \(D_i\) is coastal time minus freeway time, we typically expect \(b(X_i) < 0\).

---

## Randomization design

### Job types

Use these job types unless the user provides a different list:

```python
JOB_TYPES = [
    "data_science",
    "finance",
    "marketing",
    "product_design",
    "software_engineering",
]
```

### Release times

Conditional on job type, assign release time uniformly:

```python
RELEASE_TIMES = ["17:00", "17:15", "17:30", "17:45"]
```

Each release time has probability 0.25 within each job type.

### Weather

Weather is randomized independently of all other variables:

```python
WEATHER_STATES = ["sunny", "cloudy"]
SUNNY_PROBABILITY = 0.50
```

Weather descriptions:

```python
WEATHER_DESCRIPTIONS = {
    "sunny": (
        "It is sunny with clear evening light. The coastal route is likely to "
        "have better views, especially through Del Mar, Solana Beach, and Cardiff."
    ),
    "cloudy": (
        "It is cloudy with muted evening light. The coastal route still goes "
        "through Del Mar, Solana Beach, and Cardiff, but the scenery is less salient."
    ),
}
```

### Fatigue

Fatigue is correlated with release time but not with route-time shocks. Use this table unless the user provides a different one:

```python
FATIGUE_PROBS = {
    "17:00": {
        "refreshed": 0.35,
        "tired": 0.35,
        "mentally_worn_down": 0.20,
        "overwhelmed": 0.10,
    },
    "17:15": {
        "refreshed": 0.25,
        "tired": 0.35,
        "mentally_worn_down": 0.25,
        "overwhelmed": 0.15,
    },
    "17:30": {
        "refreshed": 0.15,
        "tired": 0.35,
        "mentally_worn_down": 0.30,
        "overwhelmed": 0.20,
    },
    "17:45": {
        "refreshed": 0.10,
        "tired": 0.25,
        "mentally_worn_down": 0.35,
        "overwhelmed": 0.30,
    },
}
```

Fatigue descriptions:

```python
FATIGUE_DESCRIPTIONS = {
    "refreshed": (
        "You feel alert and have enough energy to handle either freeway driving "
        "or local coastal driving."
    ),
    "tired": (
        "You feel tired from the workday, but you are still patient enough to "
        "make a normal route choice."
    ),
    "mentally_worn_down": (
        "You feel mentally worn down after meetings and would prefer a route "
        "that feels less stressful, all else equal."
    ),
    "overwhelmed": (
        "You feel overwhelmed and want the drive home to require as little "
        "attention and decision-making as possible."
    ),
}
```

---

## Route-time simulation

The user will provide projected times for each route and release time. The project then adds bounded heavy-tailed jitter so that travel times vary across observations while respecting physical support constraints.

For route \(r\), with distance \(d_r\), define:

\[
L_r = 60 \cdot d_r / 80
\]

\[
U_r = 60 \cdot d_r / 5
\]

where:

- \(L_r\) is the lower bound implied by driving the full route at 80 mph.
- \(U_r\) is the upper bound implied by driving the full route at 5 mph.

Let \(m_{r,t}\) be the projected time for route \(r\) at release time \(t\). Define:

\[
a_{r,t} = \frac{m_{r,t} - L_r}{U_r - L_r}
\]

\[
\eta_{r,t} = \text{logit}(a_{r,t})
\]

Draw heavy-tailed noise:

\[
\epsilon_{i,r,t} \sim s \cdot t_\nu
\]

Recommended defaults:

- \(s = 0.32\)
- \(\nu = 3\)

Then generate the prompt-visible time:

\[
T_{i,r,t} =
L_r + (U_r - L_r)\Lambda(\eta_{r,t} + \epsilon_{i,r,t})
\]

This generates bounded heavy-tailed variation around the user's provided projected time.

The script must export both base projected times and jittered prompt-visible times.

---

## Prompt template for LLM route choice

Use this prompt exactly unless the user requests edits. Fill in template variables at runtime.

```text
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
- Return valid JSON only.
```

---

## Required structured output schema

Use OpenAI structured outputs or the equivalent for the selected provider. The route-choice call must return this JSON schema:

```json
{
  "chosen_route": "freeway or coastal",
  "main_reason": "string",
  "secondary_reason": "string",
  "confidence": 0.0,
  "would_reconsider_if_difference_changed_by_minutes": 0,
  "one_sentence_summary": "string"
}
```

Enforce:

- `chosen_route` must be either `freeway` or `coastal`.
- `confidence` must be numeric between 0 and 1.
- `would_reconsider_if_difference_changed_by_minutes` must be an integer between 0 and 60.
- No extra fields.

If the model call fails, record the error and continue. Do not silently drop observations.

---

## Required dataset columns

The person-level panel must include at least:

```text
obs_id
job_type
release_time
release_time_label
fatigue_state
fatigue_description
weather
weather_description
sunny_indicator
freeway_distance_miles
coastal_distance_miles
freeway_base_minutes
coastal_base_minutes
freeway_time_minutes
coastal_time_minutes
delta_time_coastal_minus_freeway
prompt_text
chosen_route
choose_coastal
main_reason
secondary_reason
confidence
would_reconsider_if_difference_changed_by_minutes
one_sentence_summary
api_model
api_status
api_error
random_seed
```

The prompt-visible treatment variable is:

```text
delta_time_coastal_minus_freeway = coastal_time_minutes - freeway_time_minutes
```

The outcome is:

```text
choose_coastal = 1[chosen_route == "coastal"]
```

---

## Project structure to create

Build the project with this structure:

```text
commute-digital-twins/
  README.md
  requirements.txt
  pyproject.toml
  .gitignore
  .env.example

  configs/
    experiment_config.yaml

  src/
    __init__.py
    config.py
    simulate_design.py
    prompts.py
    api_client.py
    run_experiment.py
    build_design_matrix.py
    estimate_logit.py
    flm_debias.py
    targets.py
    plots.py
    report.py
    utils.py

  scripts/
    00_validate_config.py
    01_simulate_design.py
    02_run_llm_choices.py
    03_build_design_matrix.py
    04_estimate_logit.py
    05_flm_debias.py
    06_make_plots.py
    07_compile_report.py
    run_all.py

  tests/
    test_jitter.py
    test_prompt_schema.py
    test_design_matrix.py
    test_targets.py
    test_flm_influence_functions.py

  reports/
    report.tex
    report.bib
    sections/
      01_summary.tex
      02_design.tex
      03_estimation.tex
      04_results.tex
      05_linkedin_post.tex
      06_appendix.tex

  outputs/
    data/
    estimates/
    figures/
    tables/
    report/
    logs/
```

The output folder should be generated by the pipeline, not committed unless the user asks to commit outputs.

---

## Configuration file

Create `configs/experiment_config.yaml` with all user inputs and defaults.

It must include:

```yaml
experiment:
  random_seed: 12345
  n_per_job_type: null
  use_api: false

openai:
  model: null
  api_key_env_var: OPENAI_API_KEY
  max_retries: 4
  timeout_seconds: 60

routes:
  freeway_distance_miles: null
  coastal_distance_miles: null
  max_speed_mph: 80.0
  min_speed_mph: 5.0
  base_times_freeway:
    "17:00": null
    "17:15": null
    "17:30": null
    "17:45": null
  base_times_coastal:
    "17:00": null
    "17:15": null
    "17:30": null
    "17:45": null

jitter:
  df: 3
  scale: 0.32

randomization:
  job_types:
    - data_science
    - finance
    - marketing
    - product_design
    - software_engineering
  release_times:
    - "17:00"
    - "17:15"
    - "17:30"
    - "17:45"
  sunny_probability: 0.50
```

The validation script must fail clearly if required fields are still null.

---

## Estimation model

Use:

\[
p_i = \Lambda(W_i'\theta)
\]

where:

\[
W_i =
\begin{bmatrix}
A(X_i) \\
D_i B(X_i)
\end{bmatrix}
\]

with:

- \(A(X_i)\) as the flexible intercept basis.
- \(B(X_i)\) as the flexible slope basis.
- \(D_i\) as coastal-minus-freeway time differential.

A simple baseline basis should include:

- Intercept.
- Job type dummies.
- Release time dummies.
- Fatigue dummies.
- Weather dummy.
- Job type interactions with fatigue.
- Job type interactions with weather.
- Job type interactions with release time.
- Optional fatigue by weather interaction.

The slope basis should use the same or a subset of this basis interacted with \(D_i\). The script should make this configurable.

The baseline estimator should include:

1. Unregularized logistic regression if numerically stable.
2. Ridge-regularized logistic regression as default.
3. Cross-validated ridge penalty as an option.

Export all model objects needed for debiasing:

- Coefficient vector.
- Feature names.
- Fitted probabilities.
- Score contributions.
- Hessian matrix.
- Hessian contributions.
- Gradient of target functionals.
- Influence-function contributions.

---

## Score and Hessian

For each observation:

\[
\psi_i(\theta) = W_i(Y_i - p_i)
\]

where:

\[
p_i = \Lambda(W_i'\theta)
\]

The Hessian contribution is:

\[
H_i(\theta) = -p_i(1-p_i) W_iW_i'
\]

The sample Hessian is:

\[
\hat H = \frac{1}{n}\sum_i H_i(\hat\theta)
\]

Export the Hessian in both wide matrix form and long format:

```text
row_index
col_index
row_name
col_name
hessian_value
```

Also export eigenvalues and condition numbers. If regularizing the inverse Hessian, do so explicitly and export the regularization method.

---

## FLM-style debiasing layer

Implement an FLM-style debiasing layer for smooth target functionals of the estimated logit model.

### General setup

Let the model score be:

\[
E[\psi_i(\theta)] = 0
\]

where:

\[
\psi_i(\theta) = W_i(Y_i - p_i)
\]

Let:

\[
H = E[\partial \psi_i(\theta) / \partial \theta']
\]

For a target functional:

\[
\tau = E[g(Z_i, \theta)]
\]

with gradient:

\[
G = \partial E[g(Z_i, \theta)] / \partial \theta'
\]

the influence-function representation is:

\[
\phi_i =
g(Z_i, \theta) - \tau - G H^{-1}\psi_i(\theta)
\]

Use the sample analog:

\[
\hat\phi_i =
g(Z_i, \hat\theta) - \hat\tau - \hat G \hat H^{-1}\psi_i(\hat\theta)
\]

If using the sign convention \(IF_\theta = -H^{-1}\psi_i\), this expression is consistent with \(H = E[\partial \psi / \partial \theta']\). Be explicit in code comments and tests.

### Cross-fitting

Implement cross-fitting to avoid overfitting in nuisance estimation.

Minimum implementation:

- K-fold split, default K = 5.
- For each fold:
  - Estimate model and nuisance objects on training folds.
  - Evaluate fitted probabilities, scores, targets, and influence components on the held-out fold.
- Aggregate influence-function contributions across held-out folds.

If the first version uses finite-dimensional logit only, implement cross-fitting anyway so the project can later replace the logit with richer ML nuisance functions.

### Hessian inversion

Implement stable Hessian inversion options:

1. Direct inverse if condition number is acceptable.
2. Ridge regularized inverse:
   \[
   (H - \lambda I)^{-1}
   \]
   or equivalent sign-consistent version.
3. Spectral inverse with eigenvalue flooring:
   - Decompose the Hessian or information matrix.
   - Floor absolute eigenvalues at a configurable threshold.
   - Export original and floored eigenvalues.

Do not silently regularize. Every regularization choice must be logged and exported.

---

## Target parameters of interest

Implement all of these target functionals.

### 1. Average coastal choice probability

\[
\tau_p = E[p_i]
\]

Estimate overall and by job type.

### 2. Average observed coastal choice rate

\[
\tau_y = E[Y_i]
\]

Estimate overall and by job type.

### 3. Conditional slope coefficient function

Define:

\[
b(X_i) = B(X_i)'\theta_b
\]

where \(\theta_b\) are the slope-basis coefficients.

Export:

- \(b(X_i)\) for every observation.
- Mean \(E[b(X_i)]\).
- Job-specific means \(E[b(X_i) \mid job]\).
- Distribution plots of \(b(X_i)\).
- Summary quantiles of \(b(X_i)\).

### 4. Conditional intercept function

Define:

\[
a(X_i) = A(X_i)'\theta_a
\]

Export:

- \(a(X_i)\) for every observation.
- Mean \(E[a(X_i)]\).
- Job-specific means \(E[a(X_i) \mid job]\).
- Distribution plots of \(a(X_i)\).
- Summary quantiles of \(a(X_i)\).

### 5. Average marginal effect of time differential

Since:

\[
p_i = \Lambda(a(X_i) + b(X_i)D_i)
\]

the marginal effect of \(D_i\) is:

\[
ME_i = \frac{\partial p_i}{\partial D_i}
= p_i(1-p_i)b(X_i)
\]

Export:

- \(ME_i\) for every observation.
- Average marginal effect:
  \[
  AME = E[ME_i]
  \]
- Job-specific AME.
- AME by fatigue state.
- AME by weather.
- Debiased AME and standard errors.

### 6. Average semi-elasticity

The semi-elasticity of coastal choice probability with respect to minutes is:

\[
SE_i = \frac{1}{p_i}\frac{\partial p_i}{\partial D_i}
= (1-p_i)b(X_i)
\]

Export:

- \(SE_i\) for every observation.
- Overall and job-specific averages.
- Debiased estimates and standard errors.

### 7. Average elasticity with respect to the time differential

Define:

\[
EL_i =
\frac{D_i}{p_i}
\frac{\partial p_i}{\partial D_i}
=
D_i(1-p_i)b(X_i)
\]

Export:

- \(EL_i\) for every observation.
- Average elasticity:
  \[
  E[EL_i]
  \]
- Job-specific average elasticity.
- Distribution of conditional elasticities.
- Quantiles by job type.
- Debiased estimates and standard errors.

### 8. Route-time willingness to take the coastal route

Define the threshold time differential at which the model predicts indifference:

\[
p_i = 0.5
\]

This implies:

\[
a(X_i) + b(X_i)D_i^* = 0
\]

so:

\[
D_i^* = -a(X_i) / b(X_i)
\]

Interpret \(D_i^*\) as the coastal time premium, in minutes, at which the agent is indifferent between the coastal and freeway routes, conditional on \(X_i\). This is a willingness-to-take-coastal-route measure in minutes.

Handle cases where \(b(X_i)\) is near zero by trimming or flagging.

Export:

- \(D_i^*\) for every observation.
- Mean and median \(D_i^*\).
- Job-specific means and medians.
- Distribution plots.
- Trimmed estimates that exclude \(|b(X_i)| < \epsilon\).
- Sensitivity to the trimming threshold.

### 9. Sunny-day coastal premium

Compute the difference in predicted probability when weather changes from cloudy to sunny, holding all other covariates and route times fixed:

\[
\Delta p_i^{sunny} =
p(X_i, weather=sunny, D_i) -
p(X_i, weather=cloudy, D_i)
\]

Export:

- Average sunny-day effect.
- Job-specific sunny-day effect.
- Fatigue-specific sunny-day effect.
- Debiased estimate if implemented as a smooth target.

### 10. Fatigue-state effects

Compute predicted probability contrasts for fatigue states, holding other covariates fixed:

\[
p(X_i, fatigue=f, D_i) - p(X_i, fatigue=refreshed, D_i)
\]

Export contrasts for:

- tired
- mentally worn down
- overwhelmed

Do this overall and by job type.

### 11. Release-time effects

Compute predicted probability contrasts for release times, holding other covariates fixed:

\[
p(X_i, release=t, D_i) - p(X_i, release=17:00, D_i)
\]

Export overall and job-specific contrasts.

### 12. Distribution of conditional parameters

Produce clean tables and plots for:

- \(a(X_i)\), conditional intercept.
- \(b(X_i)\), conditional time slope.
- \(ME_i\), marginal effect.
- \(SE_i\), semi-elasticity.
- \(EL_i\), elasticity.
- \(D_i^*\), route-time willingness-to-take threshold.

---

## Required plots

Create high-quality PNG and PDF versions of each plot.

At minimum:

1. Route choice share by job type.
2. Route choice share by release time.
3. Route choice share by fatigue state.
4. Route choice share by weather.
5. Distribution of freeway travel time.
6. Distribution of coastal travel time.
7. Distribution of coastal-minus-freeway time differential.
8. Predicted probability curve by time differential, overall.
9. Predicted probability curve by time differential and job type.
10. Distribution of conditional slope \(b(X_i)\).
11. Distribution of conditional intercept \(a(X_i)\).
12. Distribution of marginal effects \(ME_i\).
13. Distribution of elasticities \(EL_i\).
14. Distribution of willingness-to-take thresholds \(D_i^*\).
15. Job-specific average marginal effects with confidence intervals.
16. Job-specific average elasticities with confidence intervals.
17. Sunny versus cloudy predicted probability contrast.
18. Fatigue-state predicted probability contrasts.

Use clear labels. Do not use overly playful titles in the technical plots.

---

## Required output files

Create these files:

```text
outputs/data/person_level_panel.csv
outputs/data/person_level_panel.parquet
outputs/data/prompts.jsonl
outputs/data/api_raw_responses.jsonl
outputs/data/estimation_design_matrix.csv

outputs/estimates/logit_coefficients.csv
outputs/estimates/logit_fitted_values.csv
outputs/estimates/logit_scores.csv
outputs/estimates/hessian_matrix.csv
outputs/estimates/hessian_entries_long.csv
outputs/estimates/hessian_eigenvalues.csv
outputs/estimates/target_parameter_estimates.csv
outputs/estimates/influence_function_contributions.csv
outputs/estimates/conditional_parameters.csv

outputs/tables/summary_statistics.csv
outputs/tables/route_choice_shares.csv
outputs/tables/job_type_estimates.csv
outputs/tables/fatigue_estimates.csv
outputs/tables/weather_estimates.csv

outputs/figures/*.png
outputs/figures/*.pdf

outputs/report/commute_digital_twins_report.pdf
outputs/report/commute_digital_twins_report.tex
outputs/report/linkedin_post.md

outputs/logs/run_log.txt
outputs/logs/config_resolved.yaml
```

Also create one Excel workbook:

```text
outputs/commute_digital_twins_outputs.xlsx
```

Workbook tabs:

1. `person_level_panel`
2. `summary_statistics`
3. `route_choice_shares`
4. `logit_coefficients`
5. `target_estimates`
6. `job_type_estimates`
7. `conditional_parameters`
8. `hessian_entries`
9. `influence_functions`
10. `plot_index`

---

## PDF report requirements

The report must be compiled from LaTeX. Include:

1. Title page.
2. Executive summary.
3. Experimental design.
4. Data generation and randomization.
5. Prompt used for the digital twin route-choice task.
6. Logit model.
7. FLM-style debiasing and influence-function construction.
8. Results.
9. Plots.
10. Limitations.
11. Appendix with parameter definitions.
12. LinkedIn/blog-style writeup.

The report should compile to:

```text
outputs/report/commute_digital_twins_report.pdf
```

### Report tone

The technical report should be clear, precise, and economist-readable.

The LinkedIn/blog-style section should sound like a human economist reflecting on a small digital twins experiment. It should not sound like corporate marketing copy. It should be accessible, mildly personal, and intellectually honest.

It should make these points:

- The experiment is not claiming to recover real human commuting preferences.
- It is a controlled demonstration of how digital twin agents respond to structured, randomized choice environments.
- The route choice is intentionally mundane, which makes the design easy to explain.
- The interesting object is not just the route share, but how choice changes with the time premium of the coastal route.
- The method turns a familiar daily decision into a small discrete-choice experiment.
- The results are useful as a demonstration of prompt design, simulation design, and causal/econometric structure for LLM-based agents.
- The limitations are important: these are model-implied choices, not observed human choices.

---

## LinkedIn/blog post draft to include in the report

Create `outputs/report/linkedin_post.md` and include a polished post along these lines. The exact results should be inserted after estimation.

```markdown
# Trying a digital twins experiment on my commute home

I wanted a simple way to explain digital twin experiments to people who do not spend their day thinking about choice models, orthogonal scores, or synthetic agents.

So I used one of the most ordinary decisions I make: how to drive home from work.

The setup is intentionally simple. Imagine a Wednesday evening commuter leaving the Intuit campus in Torrey Highlands and heading home to Encinitas. There are two routes. One is the standard freeway route through CA-56 and I-5. The other is the coastal route through Del Mar, Solana Beach, and Cardiff-by-the-Sea. The coastal route can be nicer, but it has traffic lights, stop signs, local traffic, and usually takes longer.

I generated a set of LLM-simulated commuters with randomized job types, release times, fatigue states, weather conditions, and route-time estimates. Each agent saw the same kind of decision a real commuter sees: two routes, two estimated travel times, and some context about the day. Then I asked the agent to choose the route it would actually take.

The point was not to claim that these choices are real human behavior. They are not. The point was to build a clean, controlled demonstration of what a digital twin experiment can look like when it is structured like a choice experiment.

The econometric object is straightforward: how does the probability of taking the coastal route change as the coastal route gets slower relative to the freeway route?

In other words, what is the time premium agents are willing to pay for the coastal route?

That turns a familiar commute decision into a small discrete-choice model. The outcome is whether the agent chooses the coastal route. The treatment is the time difference between the coastal route and the freeway route. The model lets the time sensitivity vary by observed characteristics such as job type, fatigue, weather, and release time.

What I like about this example is that the design is easy to audit. The randomization is explicit. The prompt is fixed. The route times are generated from a bounded distribution. The outputs are structured. The model estimates are interpretable.

The broader lesson is not that digital twins can tell us the true causal effect of scenic views on my commute. The lesson is that digital twin experiments need the same discipline as any other empirical design: clear treatments, clear outcomes, pre-decision covariates, careful randomization, and honest interpretation.

Even for a toy commute problem, that structure matters.
```

After actual estimation, append:

- Total number of simulated agents.
- Coastal route share.
- Average coastal time premium.
- Average marginal effect of the time differential.
- Average elasticity.
- Job-type-specific differences.
- A short limitations paragraph.

---

## Implementation notes

### API safety and secrets

Do not hard-code API keys.

Use:

```bash
export OPENAI_API_KEY="..."
```

Create `.env.example` but not `.env`.

`.gitignore` must exclude:

```text
.env
outputs/
__pycache__/
*.pyc
.ipynb_checkpoints/
```

If the user asks to commit outputs, confirm before committing because outputs may include model responses and prompts.

### Reproducibility

Every script must use fixed seeds. Write the resolved config to:

```text
outputs/logs/config_resolved.yaml
```

### Dry-run mode

The project must support dry-run mode. In dry-run mode:

- Generate the experimental design.
- Generate prompts.
- Do not call the API.
- Generate fake route choices using a simple deterministic or seeded probabilistic rule.
- Label all dry-run outputs clearly.

### API-run mode

In API mode:

- Generate the experimental design.
- Generate prompts.
- Call the API.
- Save raw responses.
- Parse structured outputs.
- Continue after failures.
- Record API errors per observation.

### Batching

If possible, implement a batch option. The synchronous version is acceptable for first build, but design the code so batch execution can be added later.

---

## Tests and validation

Implement tests for:

1. Route-time jitter respects support constraints.
2. Release time is approximately uniform within job type.
3. Weather is approximately independent of job type and release time.
4. Fatigue probabilities match the specified release-time table.
5. Prompt rendering contains all required variables.
6. Structured output parser rejects invalid route choices.
7. Design matrix has expected columns.
8. Logit score has the expected dimension.
9. Hessian matrix is square and symmetric up to numerical tolerance.
10. Influence-function contributions approximately sum to zero for each target.
11. Target functions return finite values when trimming is applied.

Also implement script-level validation:

```bash
python scripts/00_validate_config.py
pytest
```

---

## Suggested run commands

After user provides config values:

```bash
python scripts/00_validate_config.py
python scripts/01_simulate_design.py
python scripts/02_run_llm_choices.py
python scripts/03_build_design_matrix.py
python scripts/04_estimate_logit.py
python scripts/05_flm_debias.py
python scripts/06_make_plots.py
python scripts/07_compile_report.py
```

Also create:

```bash
python scripts/run_all.py
```

with flags:

```bash
python scripts/run_all.py --dry-run
python scripts/run_all.py --api-run
python scripts/run_all.py --skip-report
```

---

## GitHub instructions

After the user provides the GitHub repo and approves writing to it:

1. Clone the repo or use the provided local path.
2. Create a branch if requested.
3. Add the project files.
4. Run validation and tests.
5. Run a dry-run smoke test.
6. Commit the code with a clear commit message.
7. Push the branch if authorized.

Suggested commit message:

```text
Add digital twins commute-choice experiment pipeline
```

Do not commit secrets. Do not commit `.env`. Do not commit API keys. Do not commit large outputs unless the user explicitly requests it.

---

## Acceptance criteria

The project is complete when:

1. The repo contains the full simulation, API, estimation, plotting, and reporting pipeline.
2. The user can fill in route distances and projected route times in one config file.
3. The user can run a dry-run without API calls.
4. The user can run a real API experiment after setting `OPENAI_API_KEY`.
5. The pipeline creates a person-level dataset.
6. The pipeline creates an estimation design matrix.
7. The pipeline estimates the flexible logit model.
8. The pipeline exports coefficient estimates, Hessian entries, scores, target estimates, and influence functions.
9. The pipeline generates plots.
10. The pipeline compiles a PDF report from LaTeX.
11. The report includes both technical results and a polished LinkedIn/blog-style writeup.
12. The code is tested, reproducible, and does not hard-code secrets.
13. The project is committed and pushed to the user-provided GitHub repo if authorized.

---

## Final reminder for the coding agent

Do not overcomplicate the first implementation. Build a clean, reproducible pipeline first. Use simple finite-dimensional basis functions and ridge-regularized logit. Implement FLM-style debiasing for smooth target functionals with explicit score, Hessian, gradient, and influence-function exports. Keep the code modular so richer ML nuisances can be swapped in later.

The goal is a one-shot, end-to-end project that produces data, estimates, plots, and a report for a digital twins commute-choice experiment.

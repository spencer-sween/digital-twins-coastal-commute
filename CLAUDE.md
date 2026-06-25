# Digital Twins Commute-Choice Experiment

A fully reproducible Python pipeline that simulates Wednesday evening commuters (via Claude LLM agents) choosing between a freeway and coastal route home from the Intuit campus in Torrey Highlands to Encinitas. Estimates a flexible logit demand model with FLM-style debiased inference.

**Full spec**: `markdown/INSTRUCTIONS_CLEAN.md` | **Experiment summary**: `markdown/EXPERIMENT_SUMMARY.md`

---

## What This Project Does

1. **Simulates** person-level commute decisions across randomized job type, release time, fatigue state, weather, and jittered route times.
2. **Calls Claude** (Anthropic API) to obtain a structured JSON route choice for each synthetic agent.
3. **Estimates** `P(coastal) = Λ(a(X) + b(X)·D)` — a flexible logit where both intercept and slope vary by covariates, and `D` is the coastal-minus-freeway time differential.
4. **Debias** target functionals (AME, elasticities, willingness-to-take thresholds, fatigue/weather contrasts) via FLM-style influence functions with K-fold cross-fitting.
5. **Produces** 18+ plots, coefficient/Hessian exports, an Excel workbook, and a compiled LaTeX PDF report with a LinkedIn/blog writeup.

---

## Tech Stack

- **Language**: Python
- **LLM**: Anthropic API — `claude-sonnet-4-6` (or user-specified model)
- **Key packages**: `anthropic`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `openpyxl`, `pyyaml`, `jsonlines`, `pytest`
- **Report**: LaTeX → PDF via `pdflatex`

---

## API Setup

```bash
export ANTHROPIC_API_KEY="your-key-here"
export ANTHROPIC_MODEL="claude-sonnet-4-6"   # optional override
```

Never hard-code keys. `.env` is gitignored. Use `.env.example` as the template.

### Structured Output Pattern

Claude does not have OpenAI-style native structured outputs. Enforce JSON via a system prompt + schema validation:

```python
import anthropic, os, json, jsonschema

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM = (
    "You are participating in a route-choice study. "
    "Return ONLY valid JSON matching the schema provided. No prose, no markdown."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_route": {"type": "string", "enum": ["freeway", "coastal"]},
        "main_reason": {"type": "string"},
        "secondary_reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "would_reconsider_if_difference_changed_by_minutes": {"type": "integer", "minimum": 0, "maximum": 60},
        "one_sentence_summary": {"type": "string"},
    },
    "required": ["chosen_route", "main_reason", "secondary_reason", "confidence",
                 "would_reconsider_if_difference_changed_by_minutes", "one_sentence_summary"],
    "additionalProperties": False,
}

def call_claude(prompt_text: str, model: str, max_retries: int = 4) -> dict:
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt_text}],
            )
            data = json.loads(response.content[0].text)
            jsonschema.validate(data, SCHEMA)
            return {"status": "ok", "data": data, "error": None}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"status": "error", "data": None, "error": str(e)}
    return {"status": "error", "data": None, "error": "max retries exceeded"}
```

---

## Project Structure

```
commute-digital-twins/
  configs/experiment_config.yaml   ← fill this in before running
  src/                             ← core modules
  scripts/00_validate_config.py    ← run first
  scripts/01–07_*.py               ← numbered pipeline steps
  scripts/run_all.py               ← run everything
  tests/                           ← pytest suite (11 tests required)
  reports/report.tex               ← LaTeX source
  outputs/                         ← generated; gitignored by default
  markdown/
    INSTRUCTIONS_CLEAN.md          ← full Claude-optimized spec
    INSTRUCTIONS_DRAFT.md          ← original spec (preserved)
    EXPERIMENT_SUMMARY.md          ← phase-by-phase summary
```

---

## Run Commands

```bash
# 1. Fill in configs/experiment_config.yaml, then:
python scripts/00_validate_config.py

# Dry-run (no API calls, seeded fake choices)
python scripts/run_all.py --dry-run

# Real API run
python scripts/run_all.py --api-run

# Skip LaTeX report (if pdflatex not available)
python scripts/run_all.py --api-run --skip-report

# Tests
pytest
```

---

## Required User Inputs Before Writing Code

The coding agent must ask for these before generating any production code:

| # | Input | Notes |
|---|---|---|
| 1 | GitHub repo URL or local path | Plus: new branch? push authorization? |
| 2 | Claude model | Default: `claude-sonnet-4-6` |
| 3 | `FREEWAY_DISTANCE_MILES` | e.g. 18.5 |
| 4 | `COASTAL_DISTANCE_MILES` | e.g. 22.0 |
| 5 | Projected freeway minutes at 5:00, 5:15, 5:30, 5:45 PM | 4 values |
| 6 | Projected coastal minutes at 5:00, 5:15, 5:30, 5:45 PM | 4 values |
| 7 | Simulations per job type | Pilot: 100, Demo: 600, Precise: 1000+ |
| 8 | Start with dry-run or real API? | Recommend dry-run first |
| 9 | LinkedIn writeup author name/title | Or generic first person |

---

## Key Econometric Objects

| Symbol | Meaning |
|---|---|
| `D_i` | `coastal_time - freeway_time` (the treatment) |
| `Y_i` | `1[agent chose coastal]` (the outcome) |
| `a(X_i)` | Flexible intercept (varies by job type, fatigue, weather, release time) |
| `b(X_i)` | Flexible slope on `D_i` — time sensitivity (expect `b < 0`) |
| `ME_i` | `p_i(1-p_i)·b(X_i)` — marginal effect of time differential |
| `D_i*` | `-a(X_i)/b(X_i)` — willingness-to-take threshold (minutes of coastal premium at indifference) |

---

## GitHub Repo

`https://github.com/spencer-sween/digital-twins-coastal-commute.git`

Do not commit: `.env`, `outputs/`, API keys, large response files (unless user explicitly requests).

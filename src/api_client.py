"""
Anthropic API client with structured-output enforcement via jsonschema validation.
Supports both real API mode (USE_API=true) and dry-run mode (USE_API=false).
"""
import json
import os
import time
from typing import Optional
import numpy as np
import jsonschema
import anthropic

from src.prompts import SYSTEM_PROMPT, OUTPUT_SCHEMA


_client: Optional[anthropic.Anthropic] = None


def _get_client():  # -> anthropic.Anthropic
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def call_claude(
    prompt_text: str,
    model: str,
    max_tokens: int = 512,
    max_retries: int = 4,
    timeout: int = 60,
) -> dict:
    """Call the Anthropic API and return a validated structured response dict."""
    client = _get_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt_text}],
                timeout=timeout,
            )
            raw_text = response.content[0].text.strip()
            # Strip markdown code fences if the model added them
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1]  # drop opening fence line
                raw_text = raw_text.rsplit("```", 1)[0].strip()  # drop closing fence
            data = json.loads(raw_text)
            jsonschema.validate(data, OUTPUT_SCHEMA)
            return {"status": "ok", "data": data, "error": None}
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            last_error = f"schema_error: {e}"
        except anthropic.APIError as e:
            last_error = f"api_error: {e}"
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            last_error = f"unknown_error: {e}"

    return {"status": "error", "data": None, "error": last_error}


def _fake_choice(rng: np.random.Generator) -> dict:
    """Deterministic fake choice for dry-run mode."""
    chosen = rng.choice(["freeway", "coastal"])
    return {
        "chosen_route": chosen,
        "main_reason": "Dry-run mode: seeded fake choice.",
        "secondary_reason": "No API call was made.",
        "confidence": float(round(rng.uniform(0.5, 0.95), 2)),
        "would_reconsider_if_difference_changed_by_minutes": int(rng.integers(2, 15)),
        "one_sentence_summary": f"Chose {chosen} in dry-run mode.",
    }


def get_route_choice(
    prompt_text: str,
    cfg: dict,
    rng: np.random.Generator,
) -> dict:
    """Dispatch to real API or fake depending on config."""
    use_api = cfg["experiment"].get("use_api", False)
    if not use_api:
        data = _fake_choice(rng)
        return {"status": "dry_run", "data": data, "error": None}
    model = cfg["anthropic"]["model"]
    max_tokens = cfg["anthropic"].get("max_tokens", 512)
    max_retries = cfg["anthropic"].get("max_retries", 4)
    timeout = cfg["anthropic"].get("timeout_seconds", 60)
    return call_claude(prompt_text, model, max_tokens, max_retries, timeout)

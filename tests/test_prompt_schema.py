"""Test prompt rendering and schema validation."""
import json
import jsonschema
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import render_prompt, OUTPUT_SCHEMA


@pytest.fixture
def sample_row():
    return {
        "job_type": "data_science",
        "release_time": "17:00",
        "release_time_label": "5:00 PM",
        "fatigue_state": "tired",
        "fatigue_description": "You feel tired.",
        "weather": "sunny",
        "weather_description": "It is sunny.",
        "freeway_time_minutes": 31.5,
        "coastal_time_minutes": 42.3,
    }


def test_prompt_renders(sample_row):
    prompt = render_prompt(sample_row)
    assert "freeway" in prompt.lower()
    assert "coastal" in prompt.lower()
    assert "31.5" in prompt


def test_prompt_contains_times(sample_row):
    prompt = render_prompt(sample_row)
    assert "31.5" in prompt
    assert "42.3" in prompt


def test_schema_validates_good_output():
    good = {
        "chosen_route": "freeway",
        "main_reason": "Faster.",
        "secondary_reason": "Less traffic lights.",
        "confidence": 0.8,
        "would_reconsider_if_difference_changed_by_minutes": 5,
        "one_sentence_summary": "Chose freeway for speed.",
    }
    jsonschema.validate(good, OUTPUT_SCHEMA)  # must not raise


def test_schema_rejects_bad_route():
    bad = {
        "chosen_route": "highway",  # invalid enum
        "main_reason": "...",
        "secondary_reason": "...",
        "confidence": 0.5,
        "would_reconsider_if_difference_changed_by_minutes": 3,
        "one_sentence_summary": "...",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, OUTPUT_SCHEMA)


def test_schema_rejects_extra_fields():
    bad = {
        "chosen_route": "freeway",
        "main_reason": "...",
        "secondary_reason": "...",
        "confidence": 0.5,
        "would_reconsider_if_difference_changed_by_minutes": 3,
        "one_sentence_summary": "...",
        "extra_field": "not allowed",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, OUTPUT_SCHEMA)

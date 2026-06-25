"""Test design matrix construction."""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulate_design import simulate_design
from src.run_experiment import run_experiment
from src.build_design_matrix import build_design_matrix


@pytest.fixture
def mini_panel():
    cfg = {
        "experiment": {"random_seed": 42, "n_per_job_type": 20, "use_api": False},
        "routes": {
            "freeway_distance_miles": 14.0,
            "coastal_distance_miles": 16.0,
            "max_speed_mph": 80.0,
            "min_speed_mph": 5.0,
            "base_times_freeway": {"17:00": 31, "17:15": 31, "17:30": 31, "17:45": 29},
            "base_times_coastal": {"17:00": 42, "17:15": 42, "17:30": 41, "17:45": 40},
        },
        "jitter": {"df": 3, "scale": 0.32},
        "randomization": {
            "job_types": ["data_science", "software_engineering"],
            "release_times": ["17:00", "17:15", "17:30", "17:45"],
            "sunny_probability": 0.5,
        },
        "anthropic": {"model": "claude-haiku-4-5", "max_tokens": 512, "max_retries": 1, "timeout_seconds": 30},
        "estimation": {"ridge_alpha": 0.01, "include_interactions": True},
    }
    design = simulate_design(cfg)
    panel = run_experiment(design, cfg, verbose=False)
    return panel, cfg


def test_design_matrix_shape(mini_panel):
    panel, cfg = mini_panel
    dm = build_design_matrix(panel, cfg)
    n = panel["choose_coastal"].notna().sum()
    assert dm["W"].shape[0] == n
    assert dm["W"].shape[1] == len(dm["feature_names"])


def test_no_nan_in_W(mini_panel):
    panel, cfg = mini_panel
    dm = build_design_matrix(panel, cfg)
    assert not np.any(np.isnan(dm["W"]))


def test_D_equals_delta(mini_panel):
    panel, cfg = mini_panel
    dm = build_design_matrix(panel, cfg)
    expected = dm["df"]["delta_time_coastal_minus_freeway"].values
    np.testing.assert_allclose(dm["D"], expected, atol=1e-6)

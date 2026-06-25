"""Test that jittered travel times respect physical bounds."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulate_design import simulate_design


@pytest.fixture
def mini_cfg():
    return {
        "experiment": {"random_seed": 42, "n_per_job_type": 50},
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
            "job_types": ["data_science", "finance"],
            "release_times": ["17:00", "17:15", "17:30", "17:45"],
            "sunny_probability": 0.5,
        },
    }


def test_freeway_within_bounds(mini_cfg):
    df = simulate_design(mini_cfg)
    d = mini_cfg["routes"]["freeway_distance_miles"]
    L = 60 * d / 80
    U = 60 * d / 5
    assert (df["freeway_time_minutes"] >= L - 1e-6).all()
    assert (df["freeway_time_minutes"] <= U + 1e-6).all()


def test_coastal_within_bounds(mini_cfg):
    df = simulate_design(mini_cfg)
    d = mini_cfg["routes"]["coastal_distance_miles"]
    L = 60 * d / 80
    U = 60 * d / 5
    assert (df["coastal_time_minutes"] >= L - 1e-6).all()
    assert (df["coastal_time_minutes"] <= U + 1e-6).all()


def test_expected_n_rows(mini_cfg):
    df = simulate_design(mini_cfg)
    n_jobs = len(mini_cfg["randomization"]["job_types"])
    expected = mini_cfg["experiment"]["n_per_job_type"] * n_jobs
    assert len(df) == expected


def test_delta_computed_correctly(mini_cfg):
    df = simulate_design(mini_cfg)
    expected = (df["coastal_time_minutes"] - df["freeway_time_minutes"]).round(2)
    assert (df["delta_time_coastal_minus_freeway"] == expected).all()


def test_reproducible(mini_cfg):
    df1 = simulate_design(mini_cfg)
    df2 = simulate_design(mini_cfg)
    assert (df1["freeway_time_minutes"].values == df2["freeway_time_minutes"].values).all()

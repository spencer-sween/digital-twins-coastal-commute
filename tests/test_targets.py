"""Test target parameter computation."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulate_design import simulate_design
from src.run_experiment import run_experiment
from src.build_design_matrix import build_design_matrix
from src.estimate_logit import fit_logit
from src.flm_debias import cross_fit_logit, compute_influence_functions
from src.targets import compute_targets, targets_to_df


@pytest.fixture
def results():
    cfg = {
        "experiment": {"random_seed": 42, "n_per_job_type": 30, "use_api": False},
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
        "anthropic": {"model": "claude-haiku-4-5", "max_tokens": 512, "max_retries": 1, "timeout_seconds": 30},
        "estimation": {"ridge_alpha": 0.01, "include_interactions": False},
        "flm": {"n_folds": 3, "hessian_inversion": "ridge", "ridge_lambda": 1e-4, "spectral_floor": 1e-6},
        "event_study": {"trim_b_threshold": 0.01},
    }
    design = simulate_design(cfg)
    panel = run_experiment(design, cfg, verbose=False)
    dm = build_design_matrix(panel, cfg)
    W, Y, D = dm["W"], dm["Y"], dm["D"]
    A, B = dm["A"].values, dm["B"].values
    n_A = A.shape[1]
    fit = fit_logit(W, Y, ridge_alpha=0.01)
    cross_fit = cross_fit_logit(W, Y, cfg)
    flm = compute_influence_functions(cross_fit, W, Y, A, B, D, n_A)
    target_result = compute_targets(panel, dm, fit, flm, cfg)
    return target_result


def test_tau_p_in_range(results):
    assert 0 <= results["targets"]["tau_p"] <= 1


def test_tau_y_in_range(results):
    assert 0 <= results["targets"]["tau_y"] <= 1


def test_by_job_has_all_jobs(results):
    assert "data_science" in results["by_job"]
    assert "finance" in results["by_job"]


def test_targets_df_no_nan_values(results):
    df = targets_to_df(results)
    # At least 5 rows
    assert len(df) >= 5


def test_obs_level_has_right_columns(results):
    obs = results["obs_level"]
    for col in ["obs_id", "a_i", "b_i", "p_hat", "me_i", "dstar_i"]:
        assert col in obs.columns

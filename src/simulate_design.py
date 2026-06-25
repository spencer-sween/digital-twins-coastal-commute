"""
Generate person-level experimental observations with bounded heavy-tailed jitter.

Jitter math (logit-space):
  L_r = 60 * d_r / 80   (lower bound: fastest physically possible)
  U_r = 60 * d_r / 5    (upper bound: slowest physically possible)
  a_r,t = (m_r,t - L_r) / (U_r - L_r)
  eta_r,t = logit(a_r,t)
  eps ~ Student-t(df, scale)
  T_i,r,t = L_r + (U_r - L_r) * sigmoid(eta_r,t + eps)
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import t as student_t


FATIGUE_PROBS = {
    "17:00": {"refreshed": 0.35, "tired": 0.35, "mentally_worn_down": 0.20, "overwhelmed": 0.10},
    "17:15": {"refreshed": 0.25, "tired": 0.35, "mentally_worn_down": 0.25, "overwhelmed": 0.15},
    "17:30": {"refreshed": 0.15, "tired": 0.35, "mentally_worn_down": 0.30, "overwhelmed": 0.20},
    "17:45": {"refreshed": 0.10, "tired": 0.25, "mentally_worn_down": 0.35, "overwhelmed": 0.30},
}

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

RELEASE_TIME_LABELS = {
    "17:00": "5:00 PM",
    "17:15": "5:15 PM",
    "17:30": "5:30 PM",
    "17:45": "5:45 PM",
}


def _jitter_time(base_minutes: float, distance_miles: float, df: int, scale: float, rng: np.random.Generator) -> float:
    L = 60.0 * distance_miles / 80.0
    U = 60.0 * distance_miles / 5.0
    a = (base_minutes - L) / (U - L)
    a = float(np.clip(a, 1e-6, 1 - 1e-6))
    eta = logit(a)
    eps = float(rng.standard_t(df) * scale)
    return float(L + (U - L) * expit(eta + eps))


def simulate_design(cfg: dict) -> pd.DataFrame:
    seed = cfg["experiment"]["random_seed"]
    n = cfg["experiment"]["n_per_job_type"]
    rng = np.random.default_rng(seed)

    job_types = cfg["randomization"]["job_types"]
    release_times = cfg["randomization"]["release_times"]
    sunny_prob = cfg["randomization"]["sunny_probability"]

    fw_dist = cfg["routes"]["freeway_distance_miles"]
    co_dist = cfg["routes"]["coastal_distance_miles"]
    fw_base = cfg["routes"]["base_times_freeway"]
    co_base = cfg["routes"]["base_times_coastal"]
    df = cfg["jitter"]["df"]
    scale = cfg["jitter"]["scale"]

    rows = []
    obs_id = 0
    for job_type in job_types:
        for _ in range(n):
            release_time = rng.choice(release_times)
            fatigue_probs = FATIGUE_PROBS[release_time]
            fatigue_states = list(fatigue_probs.keys())
            fatigue_weights = list(fatigue_probs.values())
            fatigue_state = rng.choice(fatigue_states, p=fatigue_weights)
            weather = "sunny" if rng.random() < sunny_prob else "cloudy"

            fw_base_min = float(fw_base[release_time])
            co_base_min = float(co_base[release_time])
            fw_time = _jitter_time(fw_base_min, fw_dist, df, scale, rng)
            co_time = _jitter_time(co_base_min, co_dist, df, scale, rng)

            fw_rounded = round(fw_time, 2)
            co_rounded = round(co_time, 2)
            rows.append({
                "obs_id": obs_id,
                "job_type": job_type,
                "release_time": release_time,
                "release_time_label": RELEASE_TIME_LABELS[release_time],
                "fatigue_state": fatigue_state,
                "fatigue_description": FATIGUE_DESCRIPTIONS[fatigue_state],
                "weather": weather,
                "weather_description": WEATHER_DESCRIPTIONS[weather],
                "sunny_indicator": int(weather == "sunny"),
                "freeway_distance_miles": fw_dist,
                "coastal_distance_miles": co_dist,
                "freeway_base_minutes": fw_base_min,
                "coastal_base_minutes": co_base_min,
                "freeway_time_minutes": fw_rounded,
                "coastal_time_minutes": co_rounded,
                "delta_time_coastal_minus_freeway": round(co_rounded - fw_rounded, 2),
            })
            obs_id += 1

    return pd.DataFrame(rows)

"""
Phase 6: Compute and export all target parameters.

Targets:
  1. Average coastal choice probability E[p_i]
  2. Average observed choice rate E[Y_i]
  3. Conditional slope b(X_i) distribution
  4. Conditional intercept a(X_i) distribution
  5. Average marginal effect (AME) of D_i
  6. Average semi-elasticity (1-p_i)*b(X_i)
  7. Average elasticity D_i*(1-p_i)*b(X_i)
  8. Willingness-to-take threshold D_i* = -a(X_i)/b(X_i)
  9. Sunny-day coastal premium
  10. Fatigue-state contrasts
  11. Release-time contrasts
"""
import numpy as np
import pandas as pd
from scipy.special import expit


def compute_targets(panel: pd.DataFrame, dm: dict, fit: dict, flm: dict, cfg: dict) -> dict:
    df = dm["df"].copy()
    A = dm["A"].values
    B = dm["B"].values
    D = dm["D"]
    Y = dm["Y"]
    n_A = A.shape[1]

    theta = flm["theta_full"]
    theta_a = theta[:n_A]
    theta_b = theta[n_A:]

    p = flm.get("p_cf", fit["p"])
    a_i = A @ theta_a
    b_i = B @ theta_b
    me_i = p * (1 - p) * b_i
    se_i = (1 - p) * b_i
    el_i = D * (1 - p) * b_i

    trim_eps = cfg.get("event_study", {}).get("trim_b_threshold", 0.01)
    b_nz = np.abs(b_i) >= trim_eps
    dstar_i = np.where(b_nz, -a_i / b_i, np.nan)

    targets = {
        "tau_p": float(np.mean(p)),
        "tau_y": float(np.mean(Y)),
        "ame": flm.get("tau_ame_debiased", float(np.mean(me_i))),
        "se_ame": flm.get("se_ame", np.nan),
        "ame_naive": float(np.mean(me_i)),
        "avg_semi_elasticity": float(np.nanmean(se_i)),
        "avg_elasticity": float(np.nanmean(el_i)),
        "avg_dstar": float(np.nanmean(dstar_i)),
        "median_dstar": float(np.nanmedian(dstar_i)),
        "dstar_trim_n_excluded": int(np.sum(~b_nz)),
        "n_obs": len(Y),
    }

    # By job type
    by_job = {}
    for job in df["job_type"].unique():
        mask = (df["job_type"] == job).values
        by_job[job] = {
            "tau_p": float(np.mean(p[mask])),
            "tau_y": float(np.mean(Y[mask])),
            "ame": float(np.mean(me_i[mask])),
            "avg_b": float(np.mean(b_i[mask])),
            "avg_a": float(np.mean(a_i[mask])),
            "avg_dstar": float(np.nanmean(dstar_i[mask])),
        }

    # Sunny premium
    sunny = df["sunny_indicator"].values.astype(bool)
    sunny_premium = float(np.mean(p[sunny]) - np.mean(p[~sunny])) if sunny.any() and (~sunny).any() else np.nan

    # Fatigue contrasts vs refreshed
    fatigue_contrasts = {}
    for state in ["tired", "mentally_worn_down", "overwhelmed"]:
        mask_f = (df["fatigue_state"] == state).values
        mask_ref = (df["fatigue_state"] == "refreshed").values
        if mask_f.any() and mask_ref.any():
            fatigue_contrasts[state] = float(np.mean(p[mask_f]) - np.mean(p[mask_ref]))

    # Release-time contrasts vs 17:00
    release_contrasts = {}
    mask_base = (df["release_time"] == "17:00").values
    for t in ["17:15", "17:30", "17:45"]:
        mask_t = (df["release_time"] == t).values
        if mask_t.any() and mask_base.any():
            release_contrasts[t] = float(np.mean(p[mask_t]) - np.mean(p[mask_base]))

    obs_level = pd.DataFrame({
        "obs_id": df["obs_id"].values,
        "a_i": a_i,
        "b_i": b_i,
        "p_hat": p,
        "me_i": me_i,
        "se_i": se_i,
        "el_i": el_i,
        "dstar_i": dstar_i,
    })

    return {
        "targets": targets,
        "by_job": by_job,
        "sunny_premium": sunny_premium,
        "fatigue_contrasts": fatigue_contrasts,
        "release_contrasts": release_contrasts,
        "obs_level": obs_level,
    }


def targets_to_df(result: dict) -> pd.DataFrame:
    rows = []
    t = result["targets"]
    for k, v in t.items():
        rows.append({"parameter": k, "subgroup": "overall", "value": v})
    for job, vals in result["by_job"].items():
        for k, v in vals.items():
            rows.append({"parameter": k, "subgroup": f"job:{job}", "value": v})
    rows.append({"parameter": "sunny_premium", "subgroup": "overall", "value": result["sunny_premium"]})
    for state, v in result["fatigue_contrasts"].items():
        rows.append({"parameter": f"fatigue_contrast_{state}", "subgroup": "overall", "value": v})
    for t_str, v in result["release_contrasts"].items():
        rows.append({"parameter": f"release_contrast_{t_str}", "subgroup": "overall", "value": v})
    return pd.DataFrame(rows)

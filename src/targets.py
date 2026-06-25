"""
Phase 6: Compute and export all target parameters with per-job influence-function SEs.
Bonferroni z = norm.ppf(0.995) ≈ 2.576 for 5 job types.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm as scipy_norm

BONF_Z = scipy_norm.ppf(1 - 0.025 / 5)   # ≈ 2.576


def _subgroup_se(values: np.ndarray, mask: np.ndarray) -> float:
    """
    Influence-function SE for E[v_i | group] estimated via the full-sample plug-in:
      phi_i = (v_i - mu_j) * 1[i in j] / P(j)
    """
    n   = len(values)
    pj  = mask.mean()
    if pj == 0:
        return np.nan
    mu_j = values[mask].mean()
    phi  = (values - mu_j) * mask / pj
    return float(np.std(phi, ddof=1) / np.sqrt(n))


def compute_targets(panel: pd.DataFrame, dm: dict, fit: dict,
                    flm: dict, cfg: dict) -> dict:
    df   = dm["df"].copy()
    A    = dm["A"].values
    B    = dm["B"].values
    D    = dm["D"]
    Y    = dm["Y"]
    n_A  = A.shape[1]
    n    = len(Y)

    theta   = flm.get("theta_full", fit["theta"])
    theta_a = theta[:n_A]
    theta_b = theta[n_A:]

    p    = flm.get("p_cf", fit["p"])
    a_i  = A @ theta_a
    b_i  = B @ theta_b
    me_i = p * (1 - p) * b_i
    se_i = (1 - p) * b_i
    el_i = D * (1 - p) * b_i

    trim_eps  = float(cfg.get("event_study", {}).get("trim_b_threshold", 0.01))
    b_nz      = np.abs(b_i) >= trim_eps
    dstar_i   = np.where(b_nz, -a_i / b_i, np.nan)

    # ── Overall targets ──────────────────────────────────────────────────────
    targets = {
        "tau_p":             float(np.mean(p)),
        "tau_y":             float(np.mean(Y)),
        "ame":               flm.get("tau_ame_debiased", float(np.mean(me_i))),
        "se_ame":            flm.get("se_ame", np.nan),
        "ame_naive":         float(np.mean(me_i)),
        "avg_semi_elasticity": float(np.nanmean(se_i)),
        "avg_elasticity":    float(np.nanmean(el_i)),
        "avg_dstar":         float(np.nanmean(dstar_i)),
        "median_dstar":      float(np.nanmedian(dstar_i)),
        "dstar_trim_n_excluded": int(np.sum(~b_nz)),
        "n_obs":             int(n),
    }

    # ── Per-job targets with IF-based SEs ───────────────────────────────────
    by_job = {}
    jobs   = df["job_type"].unique()
    for job in jobs:
        mask = (df["job_type"] == job).values.astype(float)
        mask_b = mask.astype(bool)

        # means
        tau_p_j  = float(p[mask_b].mean())
        tau_y_j  = float(Y[mask_b].mean())
        ame_j    = float(me_i[mask_b].mean())
        el_j_arr = el_i[mask_b]
        el_j_arr = el_j_arr[np.isfinite(el_j_arr)]
        el_j     = float(np.mean(el_j_arr)) if len(el_j_arr) else np.nan
        ds_j_arr = dstar_i[mask_b]
        ds_j_arr = ds_j_arr[np.isfinite(ds_j_arr)]
        ds_j     = float(np.mean(ds_j_arr)) if len(ds_j_arr) else np.nan

        # SEs via influence function
        se_p_j   = _subgroup_se(p,   mask_b)
        se_y_j   = _subgroup_se(Y.astype(float), mask_b)
        se_ame_j = _subgroup_se(me_i, mask_b)

        el_full  = np.where(np.isfinite(el_i), el_i, np.nanmedian(el_i))
        se_el_j  = _subgroup_se(el_full, mask_b)

        ds_full  = np.where(np.isfinite(dstar_i), dstar_i, np.nanmedian(dstar_i))
        se_ds_j  = _subgroup_se(ds_full, mask_b)

        by_job[job] = {
            "tau_p": tau_p_j,  "se_tau_p": se_p_j,
            "tau_y": tau_y_j,  "se_tau_y": se_y_j,
            "ame":   ame_j,    "se_ame":   se_ame_j,
            "avg_elasticity": el_j,  "se_elasticity": se_el_j,
            "avg_dstar":      ds_j,  "se_dstar":      se_ds_j,
            "n": int(mask_b.sum()),
        }

    # ── Other contrasts ──────────────────────────────────────────────────────
    sunny = df["sunny_indicator"].values.astype(bool)
    sunny_premium = float(np.mean(p[sunny]) - np.mean(p[~sunny])) \
        if sunny.any() and (~sunny).any() else np.nan

    fatigue_contrasts = {}
    mask_ref = (df["fatigue_state"] == "refreshed").values
    for state in ["tired", "mentally_worn_down", "overwhelmed"]:
        mask_f = (df["fatigue_state"] == state).values
        if mask_f.any() and mask_ref.any():
            fatigue_contrasts[state] = float(np.mean(p[mask_f]) - np.mean(p[mask_ref]))

    release_contrasts = {}
    mask_base = (df["release_time"] == "17:00").values
    for t in ["17:15", "17:30", "17:45"]:
        mask_t = (df["release_time"] == t).values
        if mask_t.any() and mask_base.any():
            release_contrasts[t] = float(np.mean(p[mask_t]) - np.mean(p[mask_base]))

    obs_level = pd.DataFrame({
        "obs_id":  df["obs_id"].values,
        "a_i":     a_i,
        "b_i":     b_i,
        "p_hat":   p,
        "me_i":    me_i,
        "se_i":    se_i,
        "el_i":    el_i,
        "dstar_i": dstar_i,
    })

    return {
        "targets":           targets,
        "by_job":            by_job,
        "sunny_premium":     sunny_premium,
        "fatigue_contrasts": fatigue_contrasts,
        "release_contrasts": release_contrasts,
        "obs_level":         obs_level,
        "bonf_z":            BONF_Z,
    }


def targets_to_df(result: dict) -> pd.DataFrame:
    rows = []
    for k, v in result["targets"].items():
        rows.append({"parameter": k, "subgroup": "overall", "value": v, "se": np.nan})
    for job, vals in result["by_job"].items():
        for k in ["tau_p", "tau_y", "ame", "avg_elasticity", "avg_dstar"]:
            rows.append({
                "parameter": k,
                "subgroup":  f"job:{job}",
                "value":     vals.get(k, np.nan),
                "se":        vals.get(f"se_{k.replace('avg_','')}", np.nan),
            })
    rows.append({"parameter": "sunny_premium", "subgroup": "overall",
                 "value": result["sunny_premium"], "se": np.nan})
    for s, v in result["fatigue_contrasts"].items():
        rows.append({"parameter": f"fatigue_contrast_{s}", "subgroup": "overall",
                     "value": v, "se": np.nan})
    for t, v in result["release_contrasts"].items():
        rows.append({"parameter": f"release_contrast_{t}", "subgroup": "overall",
                     "value": v, "se": np.nan})
    return pd.DataFrame(rows)

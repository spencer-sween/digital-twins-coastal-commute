"""
Phase 3: Build estimation-ready design matrix.

The flexible logit index is:
  W_i' theta = A(X_i)' theta_a + D_i * B(X_i)' theta_b

A(X_i) — intercept basis (flexible intercept function)
B(X_i) — slope basis interacted with D_i (flexible slope function)
"""
import numpy as np
import pandas as pd


def _job_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["job_type"], prefix="job", drop_first=True).astype(float)


def _release_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["release_time"], prefix="rel", drop_first=True).astype(float)


def _fatigue_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df["fatigue_state"], prefix="fat", drop_first=True).astype(float)


def build_intercept_basis(df: pd.DataFrame, include_interactions: bool = True) -> pd.DataFrame:
    """Build A(X_i) — the flexible intercept basis."""
    parts = [pd.DataFrame({"intercept": np.ones(len(df))})]
    job_d = _job_dummies(df)
    rel_d = _release_dummies(df)
    fat_d = _fatigue_dummies(df)
    weather_d = pd.DataFrame({"sunny": df["sunny_indicator"].values.astype(float)})

    parts += [job_d, rel_d, fat_d, weather_d]

    if include_interactions:
        # job x fatigue
        for jc in job_d.columns:
            for fc in fat_d.columns:
                col_name = f"{jc}_x_{fc}"
                parts.append(pd.DataFrame({col_name: (df[jc.replace("job_", "job_type") == df["job_type"]].values if False else (job_d[jc] * fat_d[fc]).values)}))
        # job x weather
        for jc in job_d.columns:
            parts.append(pd.DataFrame({f"{jc}_x_sunny": (job_d[jc] * weather_d["sunny"]).values}))
        # job x release_time
        for jc in job_d.columns:
            for rc in rel_d.columns:
                parts.append(pd.DataFrame({f"{jc}_x_{rc}": (job_d[jc] * rel_d[rc]).values}))

    A = pd.concat(parts, axis=1)
    A.index = df.index
    return A


def build_slope_basis(df: pd.DataFrame) -> pd.DataFrame:
    """Build B(X_i) — slope basis (reuse intercept basis without constant)."""
    parts = [pd.DataFrame({"slope_intercept": np.ones(len(df))})]
    job_d = _job_dummies(df)
    rel_d = _release_dummies(df)
    fat_d = _fatigue_dummies(df)
    weather_d = pd.DataFrame({"slope_sunny": df["sunny_indicator"].values.astype(float)})
    parts += [
        job_d.rename(columns={c: f"slope_{c}" for c in job_d.columns}),
        rel_d.rename(columns={c: f"slope_{c}" for c in rel_d.columns}),
        fat_d.rename(columns={c: f"slope_{c}" for c in fat_d.columns}),
        weather_d,
    ]
    B = pd.concat(parts, axis=1)
    B.index = df.index
    return B


def build_design_matrix(panel: pd.DataFrame, cfg: dict) -> dict:
    """Return full estimation design matrix and metadata."""
    include_ix = cfg.get("estimation", {}).get("include_interactions", True)
    df = panel.dropna(subset=["choose_coastal"]).copy().reset_index(drop=True)
    D = df["delta_time_coastal_minus_freeway"].values.astype(float)
    Y = df["choose_coastal"].values.astype(float)

    A = build_intercept_basis(df, include_ix)
    B = build_slope_basis(df)

    # W_i = [A(X_i); D_i * B(X_i)]
    D_B = B.multiply(D, axis=0)
    D_B.columns = [f"D_x_{c}" for c in B.columns]

    W = pd.concat([A, D_B], axis=1)

    return {
        "W": W.values.astype(float),
        "Y": Y,
        "D": D,
        "A": A,
        "B": B,
        "W_df": W,
        "feature_names": list(W.columns),
        "intercept_names": list(A.columns),
        "slope_names": list(B.columns),
        "df": df,
    }

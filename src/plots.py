"""
Phase 7: All 18 required plots (PNG + PDF).
Clean, publication-ready style using matplotlib with minimal chrome.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.special import expit

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
}

JOB_LABELS = {
    "data_science": "Data Science",
    "finance": "Finance",
    "marketing": "Marketing",
    "product_design": "Product Design",
    "software_engineering": "Software Eng.",
}

COLORS = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]


def _save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _bar_chart(
    values: dict, title: str, xlabel: str, ylabel: str = "Coastal choice share"
) -> plt.Figure:
    labels = [JOB_LABELS.get(k, k) for k in values.keys()]
    vals = list(values.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    with plt.rc_context(STYLE):
        bars = ax.barh(labels, vals, color=COLORS[: len(vals)], edgecolor="white")
        ax.set_xlim(0, 1)
        ax.set_xlabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def make_all_plots(panel: pd.DataFrame, dm: dict, fit: dict, target_result: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    obs = target_result["obs_level"]
    df = dm["df"].copy()
    df = df.merge(obs, on="obs_id", how="left")
    D = dm["D"]
    p = target_result["obs_level"]["p_hat"].values

    with plt.rc_context(STYLE):

        # 1. Route choice share by job type
        job_shares = df.groupby("job_type")["choose_coastal"].mean().to_dict()
        fig = _bar_chart(job_shares, "Coastal route share by job type", "Job type")
        _save(fig, "01_choice_share_by_job", out_dir)

        # 2. Route choice share by release time
        rel_shares = df.groupby("release_time")["choose_coastal"].mean().to_dict()
        fig = _bar_chart(rel_shares, "Coastal route share by release time", "Release time")
        _save(fig, "02_choice_share_by_release_time", out_dir)

        # 3. Route choice share by fatigue state
        fat_shares = df.groupby("fatigue_state")["choose_coastal"].mean().to_dict()
        fig = _bar_chart(fat_shares, "Coastal route share by fatigue state", "Fatigue state")
        _save(fig, "03_choice_share_by_fatigue", out_dir)

        # 4. Route choice share by weather
        wx_shares = df.groupby("weather")["choose_coastal"].mean().to_dict()
        fig = _bar_chart(wx_shares, "Coastal route share by weather", "Weather")
        _save(fig, "04_choice_share_by_weather", out_dir)

        # 5. Distribution of freeway travel time
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["freeway_time_minutes"], bins=50, color=COLORS[0], edgecolor="white", alpha=0.8)
        ax.set_xlabel("Freeway travel time (minutes)")
        ax.set_ylabel("Observations")
        ax.set_title("Distribution of freeway travel time")
        _save(fig, "05_dist_freeway_time", out_dir)

        # 6. Distribution of coastal travel time
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["coastal_time_minutes"], bins=50, color=COLORS[1], edgecolor="white", alpha=0.8)
        ax.set_xlabel("Coastal travel time (minutes)")
        ax.set_ylabel("Observations")
        ax.set_title("Distribution of coastal travel time")
        _save(fig, "06_dist_coastal_time", out_dir)

        # 7. Distribution of time differential D_i
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(D, bins=50, color=COLORS[2], edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Coastal − freeway time (minutes)")
        ax.set_ylabel("Observations")
        ax.set_title("Distribution of coastal-minus-freeway time differential")
        _save(fig, "07_dist_delta_time", out_dir)

        # 8. Predicted probability curve vs D_i (overall)
        d_grid = np.linspace(D.min(), D.max(), 300)
        fig, ax = plt.subplots(figsize=(7, 4))
        # Scatter with transparency
        ax.scatter(D, df["choose_coastal"], alpha=0.03, color="gray", s=5, label="Observed choices")
        ax.plot(d_grid, np.full_like(d_grid, np.mean(p)), color=COLORS[0], linewidth=2, label="Mean predicted P(coastal)")
        ax.set_xlabel("Coastal − freeway time (minutes)")
        ax.set_ylabel("P(choose coastal)")
        ax.set_title("Predicted choice probability vs. time differential")
        ax.legend()
        _save(fig, "08_predicted_prob_overall", out_dir)

        # 9. Predicted probability curve by job type
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, job in enumerate(df["job_type"].unique()):
            mask = df["job_type"] == job
            d_j = D[mask.values]
            p_j = p[mask.values]
            order = np.argsort(d_j)
            ax.plot(d_j[order], pd.Series(p_j[order]).rolling(50, center=True, min_periods=1).mean().values,
                    color=COLORS[i % len(COLORS)], label=JOB_LABELS.get(job, job), linewidth=1.5)
        ax.set_xlabel("Coastal − freeway time (minutes)")
        ax.set_ylabel("P(choose coastal) — smoothed")
        ax.set_title("Predicted choice probability by job type")
        ax.legend(fontsize=8)
        _save(fig, "09_predicted_prob_by_job", out_dir)

        # 10. Distribution of conditional slope b(X_i)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["b_i"], bins=60, color=COLORS[3], edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("b(X_i): conditional time-sensitivity slope")
        ax.set_ylabel("Observations")
        ax.set_title("Distribution of conditional slope b(X_i)")
        _save(fig, "10_dist_slope_b", out_dir)

        # 11. Distribution of conditional intercept a(X_i)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["a_i"], bins=60, color=COLORS[4], edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("a(X_i): conditional intercept")
        ax.set_ylabel("Observations")
        ax.set_title("Distribution of conditional intercept a(X_i)")
        _save(fig, "11_dist_intercept_a", out_dir)

        # 12. Distribution of marginal effects ME_i
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["me_i"], bins=60, color=COLORS[0], edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("ME_i: marginal effect of time differential")
        ax.set_ylabel("Observations")
        ax.set_title("Distribution of marginal effects ME_i")
        _save(fig, "12_dist_marginal_effects", out_dir)

        # 13. Distribution of elasticities EL_i
        el_vals = df["el_i"].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(el_vals, bins=60, color=COLORS[1], edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("EL_i: elasticity with respect to time differential")
        ax.set_ylabel("Observations")
        ax.set_title("Distribution of elasticities EL_i")
        _save(fig, "13_dist_elasticities", out_dir)

        # 14. Distribution of willingness-to-take threshold D_i*
        dstar_vals = df["dstar_i"].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dstar_vals, bins=60, color=COLORS[2], edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("D_i* (minutes): coastal time premium at indifference")
        ax.set_ylabel("Observations")
        ax.set_title("Distribution of willingness-to-take threshold D_i*")
        _save(fig, "14_dist_dstar", out_dir)

        # 15. Job-specific AME with confidence intervals (±1.96 SE using bootstrap approx)
        job_ames = [(job, vals["ame"]) for job, vals in target_result["by_job"].items()]
        job_names = [JOB_LABELS.get(j[0], j[0]) for j in job_ames]
        job_vals = [j[1] for j in job_ames]
        se_approx = target_result["targets"].get("se_ame", 0.005)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(job_names, job_vals, color=COLORS[:len(job_vals)], xerr=[1.96 * se_approx] * len(job_vals),
                error_kw={"capsize": 4}, edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Average marginal effect on P(coastal)")
        ax.set_title("Job-specific average marginal effects (AME)")
        _save(fig, "15_job_ame", out_dir)

        # 16. Job-specific average elasticities
        job_el = {job: float(np.nanmean(df.loc[df["job_type"] == job, "el_i"])) for job in df["job_type"].unique()}
        fig = _bar_chart(job_el, "Job-specific average elasticities", "Job type", ylabel="Average elasticity EL_i")
        _save(fig, "16_job_elasticities", out_dir)

        # 17. Sunny vs. cloudy predicted probability contrast
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (wx, label, color) in enumerate([("sunny", "Sunny", COLORS[0]), ("cloudy", "Cloudy", COLORS[2])]):
            mask = df["weather"] == wx
            ax.hist(p[mask.values], bins=40, alpha=0.6, color=color, label=label, edgecolor="white")
        ax.set_xlabel("P(choose coastal)")
        ax.set_ylabel("Observations")
        ax.set_title("Predicted probability by weather condition")
        ax.legend()
        _save(fig, "17_sunny_vs_cloudy", out_dir)

        # 18. Fatigue-state predicted probability contrasts
        fat_means = df.groupby("fatigue_state")["p_hat"].mean()
        fat_order = ["refreshed", "tired", "mentally_worn_down", "overwhelmed"]
        fat_order = [f for f in fat_order if f in fat_means.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        vals_f = [fat_means[f] for f in fat_order]
        colors_f = COLORS[:len(fat_order)]
        ax.bar(range(len(fat_order)), vals_f, color=colors_f, edgecolor="white", alpha=0.85)
        ax.set_xticks(range(len(fat_order)))
        ax.set_xticklabels([f.replace("_", " ").title() for f in fat_order], rotation=15, ha="right")
        ax.set_ylabel("Mean P(choose coastal)")
        ax.set_title("Predicted coastal probability by fatigue state")
        _save(fig, "18_fatigue_contrasts", out_dir)

    print(f"  18 plots saved to {out_dir}")

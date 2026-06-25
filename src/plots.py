"""
All plots for the Digital Twins Commute-Choice Experiment.
Professional publication-ready style throughout.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, norm as scipy_norm
from pathlib import Path

# ── Palette & style ──────────────────────────────────────────────────────────
PALETTE = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED"]
GRAY    = "#6B7280"
LIGHT   = "#F3F4F6"

JOB_LABELS = {
    "data_science":       "Data Science",
    "finance":            "Finance",
    "marketing":          "Marketing",
    "product_design":     "Product Design",
    "software_engineering": "Software Eng.",
}
JOB_ORDER = list(JOB_LABELS.keys())

FATIGUE_ORDER  = ["refreshed", "tired", "mentally_worn_down", "overwhelmed"]
FATIGUE_LABELS = {
    "refreshed":          "Refreshed",
    "tired":              "Tired",
    "mentally_worn_down": "Mentally Worn Down",
    "overwhelmed":        "Overwhelmed",
}

BASE_STYLE = {
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.color":          "#E5E7EB",
    "grid.linewidth":      0.7,
    "axes.axisbelow":      True,
    "font.family":         "sans-serif",
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.titleweight":    "bold",
    "axes.labelsize":      10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#E5E7EB",
}

BONF_Z = scipy_norm.ppf(1 - 0.025 / 5)   # Bonferroni for 5 job types ≈ 2.576


def _save(fig, name: str, out_dir: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _hbar(ax, labels, values, errors=None, palette=None, fmt=".1%", xmax=1.0):
    palette = palette or PALETTE
    colors  = (palette * ((len(labels) // len(palette)) + 1))[:len(labels)]
    bars = ax.barh(labels, values, xerr=errors,
                   color=colors, edgecolor="white", linewidth=0.5,
                   error_kw={"capsize": 3, "elinewidth": 1, "ecolor": GRAY},
                   zorder=3, height=0.6)
    ax.set_xlim(0, xmax * 1.12)
    if fmt == ".1%":
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    for bar, v in zip(bars, values):
        x = v + (errors[bars.patches.index(bar)] if errors else 0) + xmax * 0.01
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                (f"{v:{fmt}}" if fmt != ".1%" else f"{v:.1%}"),
                va="center", fontsize=8.5, color="#374151")
    ax.invert_yaxis()


def _kde_hist(ax, data, color, label=None, bins=30, bw=None):
    data = np.array(data)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        return
    ax.hist(data, bins=bins, density=True, color=color,
            alpha=0.25, edgecolor="white", linewidth=0.3, zorder=2)
    try:
        kde = gaussian_kde(data, bw_method=bw or "scott")
        x   = np.linspace(data.min() - 0.1 * np.ptp(data),
                          data.max() + 0.1 * np.ptp(data), 300)
        ax.plot(x, kde(x), color=color, linewidth=2, zorder=3, label=label)
    except Exception:
        pass
    ax.axvline(np.mean(data), color=color, linewidth=1.2,
               linestyle="--", alpha=0.8, zorder=4)


# ── Individual plot functions ─────────────────────────────────────────────────

def _p01_choice_by_job(df, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        jobs    = [j for j in JOB_ORDER if j in df["job_type"].unique()]
        vals    = [df.loc[df["job_type"] == j, "choose_coastal"].mean() for j in jobs]
        labels  = [JOB_LABELS[j] for j in jobs]
        n_j     = [int((df["job_type"] == j).sum()) for j in jobs]
        se      = [np.sqrt(v * (1 - v) / n) for v, n in zip(vals, n_j)]
        errs    = [BONF_Z * s for s in se]
        _hbar(ax, labels, vals, errs)
        ax.set_xlabel("Coastal route choice share")
        ax.set_title("Coastal Route Choice Share by Job Type  (95% CI, Bonferroni)")
        ax.axvline(df["choose_coastal"].mean(), color=GRAY,
                   linewidth=1, linestyle=":", label=f"Overall: {df['choose_coastal'].mean():.1%}")
        ax.legend()
        _save(fig, "01_choice_share_by_job", out_dir)


def _p02_choice_by_release(df, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.2))
        times  = ["17:00", "17:15", "17:30", "17:45"]
        labels = ["5:00 PM", "5:15 PM", "5:30 PM", "5:45 PM"]
        vals   = [df.loc[df["release_time"] == t, "choose_coastal"].mean() for t in times]
        n_t    = [int((df["release_time"] == t).sum()) for t in times]
        se     = [np.sqrt(v * (1 - v) / n) for v, n in zip(vals, n_t)]
        _hbar(ax, labels, vals, [1.96 * s for s in se])
        ax.set_xlabel("Coastal route choice share")
        ax.set_title("Coastal Route Choice Share by Release Time  (95% CI)")
        _save(fig, "02_choice_share_by_release_time", out_dir)


def _p03_choice_by_fatigue(df, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fats   = [f for f in FATIGUE_ORDER if f in df["fatigue_state"].unique()]
        labels = [FATIGUE_LABELS[f] for f in fats]
        vals   = [df.loc[df["fatigue_state"] == f, "choose_coastal"].mean() for f in fats]
        n_f    = [int((df["fatigue_state"] == f).sum()) for f in fats]
        se     = [np.sqrt(v * (1 - v) / n) for v, n in zip(vals, n_f)]
        _hbar(ax, labels, vals, [1.96 * s for s in se])
        ax.set_xlabel("Coastal route choice share")
        ax.set_title("Coastal Route Choice Share by Fatigue State  (95% CI)")
        _save(fig, "03_choice_share_by_fatigue", out_dir)


def _p04_choice_by_weather(df, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(5, 2.8))
        weather = ["sunny", "cloudy"]
        vals    = [df.loc[df["weather"] == w, "choose_coastal"].mean() for w in weather]
        n_w     = [int((df["weather"] == w).sum()) for w in weather]
        se      = [np.sqrt(v * (1 - v) / n) for v, n in zip(vals, n_w)]
        _hbar(ax, ["Sunny", "Cloudy"], vals, [1.96 * s for s in se],
              palette=["#F59E0B", "#60A5FA"])
        ax.set_xlabel("Coastal route choice share")
        ax.set_title("Coastal Route Choice Share by Weather  (95% CI)")
        _save(fig, "04_choice_share_by_weather", out_dir)


def _p05_dist_freeway(df, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _kde_hist(ax, df["freeway_time_minutes"], PALETTE[0])
        ax.set_xlabel("Freeway travel time (minutes)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Freeway Travel Time")
        _save(fig, "05_dist_freeway_time", out_dir)


def _p06_dist_coastal(df, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _kde_hist(ax, df["coastal_time_minutes"], PALETTE[1])
        ax.set_xlabel("Coastal travel time (minutes)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Coastal Travel Time")
        _save(fig, "06_dist_coastal_time", out_dir)


def _p07_dist_delta(df, D, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _kde_hist(ax, D, PALETTE[2])
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Coastal − Freeway travel time (minutes)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Time Differential  $D_i$")
        ax.text(0.97, 0.92, f"Mean: {np.mean(D):.1f} min",
                transform=ax.transAxes, ha="right", fontsize=9, color=GRAY)
        _save(fig, "07_dist_delta_time", out_dir)


def _p08_predicted_prob_overall(df, D, p, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(D, df["choose_coastal"], alpha=0.06, color=GRAY,
                   s=10, zorder=1, label="Observed choice (jittered)")
        order = np.argsort(D)
        smooth = pd.Series(p[order]).rolling(40, center=True, min_periods=1).mean()
        ax.plot(D[order], smooth.values, color=PALETTE[0],
                linewidth=2.5, zorder=3, label="Smoothed P(coastal)")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Coastal − Freeway time differential  $D_i$ (minutes)")
        ax.set_ylabel("P(choose coastal)")
        ax.set_title("Predicted Coastal Choice Probability vs. Time Differential")
        ax.legend()
        _save(fig, "08_predicted_prob_overall", out_dir)


def _p09_predicted_prob_by_job(df, D, p, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        jobs = [j for j in JOB_ORDER if j in df["job_type"].unique()]
        for i, job in enumerate(jobs):
            mask  = (df["job_type"] == job).values
            d_j   = D[mask]
            p_j   = p[mask]
            order = np.argsort(d_j)
            smooth = pd.Series(p_j[order]).rolling(20, center=True, min_periods=1).mean()
            ax.plot(d_j[order], smooth.values,
                    color=PALETTE[i % len(PALETTE)],
                    label=JOB_LABELS[job], linewidth=2, zorder=3)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Coastal − Freeway time differential  $D_i$ (minutes)")
        ax.set_ylabel("P(choose coastal)  [smoothed]")
        ax.set_title("Predicted Choice Probability by Job Type")
        ax.legend(framealpha=0.95)
        _save(fig, "09_predicted_prob_by_job", out_dir)


def _p10_dist_slope(obs, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _kde_hist(ax, obs["b_i"], PALETTE[3])
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Conditional time-sensitivity slope  $b(X_i)$")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Conditional Slope  $b(X_i)$")
        ax.text(0.03, 0.92, f"Mean: {obs['b_i'].mean():.4f}",
                transform=ax.transAxes, fontsize=9, color=GRAY)
        _save(fig, "10_dist_slope_b", out_dir)


def _p11_dist_intercept(obs, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _kde_hist(ax, obs["a_i"], PALETTE[4])
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Conditional intercept  $a(X_i)$")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Conditional Intercept  $a(X_i)$")
        ax.text(0.97, 0.92, f"Mean: {obs['a_i'].mean():.4f}",
                transform=ax.transAxes, ha="right", fontsize=9, color=GRAY)
        _save(fig, "11_dist_intercept_a", out_dir)


def _p12_dist_me(obs, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _kde_hist(ax, obs["me_i"], PALETTE[0])
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Marginal effect of  $D_i$  on P(coastal)  [i.e. $ME_i$]")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Individual Marginal Effects  $ME_i$")
        _save(fig, "12_dist_marginal_effects", out_dir)


def _p13_dist_elasticity(obs, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        el = obs["el_i"].replace([np.inf, -np.inf], np.nan).dropna()
        q1, q99 = el.quantile(0.01), el.quantile(0.99)
        el_trim = el[(el >= q1) & (el <= q99)]
        _kde_hist(ax, el_trim, PALETTE[1])
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Elasticity of coastal choice w.r.t. $D_i$  ($EL_i$)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Elasticities  $EL_i$  [1st–99th pct trimmed]")
        ax.text(0.97, 0.92, f"Mean: {el.mean():.3f}",
                transform=ax.transAxes, ha="right", fontsize=9, color=GRAY)
        _save(fig, "13_dist_elasticities", out_dir)


def _p14_dist_dstar(obs, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ds = obs["dstar_i"].replace([np.inf, -np.inf], np.nan).dropna()
        q5, q95 = ds.quantile(0.05), ds.quantile(0.95)
        ds_trim = ds[(ds >= q5) & (ds <= q95)]
        _kde_hist(ax, ds_trim, PALETTE[2])
        ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel("Coastal time premium at indifference  $D_i^*$ (minutes)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Willingness-to-Take Threshold  $D_i^*$\n[5th–95th pct trimmed]")
        ax.text(0.97, 0.92, f"Median: {ds.median():.1f} min",
                transform=ax.transAxes, ha="right", fontsize=9, color=GRAY)
        _save(fig, "14_dist_dstar", out_dir)


def _p15_job_ame(df, obs, se_ame, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        jobs   = [j for j in JOB_ORDER if j in df["job_type"].unique()]
        labels = [JOB_LABELS[j] for j in jobs]
        n      = len(df)
        vals, errs = [], []
        for job in jobs:
            mask = (df["job_type"] == job).values
            me_j = obs.loc[obs["obs_id"].isin(df.loc[mask, "obs_id"]), "me_i"]
            pj   = mask.mean()
            phi  = (me_j.values - me_j.mean()) * mask[mask] / pj if False else \
                   ((obs["me_i"].values - obs["me_i"].values[mask].mean()) * mask / pj)
            se_j = np.std(phi, ddof=1) / np.sqrt(n)
            vals.append(float(me_j.mean()))
            errs.append(BONF_Z * se_j)
        _hbar(ax, labels, vals, errs, fmt=".4f", xmax=max(abs(v) for v in vals) * 1.3 or 0.01)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Average marginal effect on P(coastal)")
        ax.set_title(f"Job-Specific Average Marginal Effects  (95% CI, Bonferroni)")
        _save(fig, "15_job_ame", out_dir)


def _p16_job_elasticity(df, obs, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        jobs   = [j for j in JOB_ORDER if j in df["job_type"].unique()]
        labels = [JOB_LABELS[j] for j in jobs]
        n      = len(obs)
        vals, errs = [], []
        for job in jobs:
            mask = (df["job_type"] == job).values
            el_j = obs["el_i"].values[mask]
            el_j = el_j[np.isfinite(el_j)]
            pj   = mask.mean()
            phi  = (obs["el_i"].fillna(obs["el_i"].median()).values - np.nanmean(el_j)) * mask / pj
            se_j = np.std(phi, ddof=1) / np.sqrt(n)
            vals.append(float(np.nanmean(el_j)))
            errs.append(BONF_Z * se_j)
        _hbar(ax, labels, vals, errs, fmt=".3f",
              xmax=max(abs(v) for v in vals) * 1.4 or 0.1)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Average elasticity of coastal choice w.r.t. $D_i$")
        ax.set_title("Job-Specific Average Elasticity  (95% CI, Bonferroni)")
        _save(fig, "16_job_elasticities", out_dir)


def _p17_sunny_cloudy(df, obs, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for wx, label, color in [("sunny", "Sunny", "#F59E0B"), ("cloudy", "Cloudy", "#60A5FA")]:
            mask = (df["weather"] == wx).values
            _kde_hist(ax, obs["p_hat"].values[mask], color, label=label)
        ax.set_xlabel("Predicted P(choose coastal)")
        ax.set_ylabel("Density")
        ax.set_title("Predicted Coastal Probability by Weather Condition")
        ax.legend()
        _save(fig, "17_sunny_vs_cloudy", out_dir)


def _p18_fatigue_contrasts(df, obs, out_dir):
    with plt.rc_context(BASE_STYLE):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fats   = [f for f in FATIGUE_ORDER if f in df["fatigue_state"].unique()]
        labels = [FATIGUE_LABELS[f] for f in fats]
        vals   = [obs["p_hat"].values[(df["fatigue_state"] == f).values].mean() for f in fats]
        ax.bar(range(len(fats)), vals,
               color=PALETTE[:len(fats)], edgecolor="white", linewidth=0.5,
               zorder=3, width=0.6)
        ax.set_xticks(range(len(fats)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Mean predicted P(choose coastal)")
        ax.set_title("Predicted Coastal Probability by Fatigue State")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, color="#374151")
        _save(fig, "18_fatigue_contrasts", out_dir)


def _p19_job_elasticity_dist(df, obs, out_dir):
    """Per-job KDE of individual elasticities — 5-panel figure."""
    with plt.rc_context(BASE_STYLE):
        jobs = [j for j in JOB_ORDER if j in df["job_type"].unique()]
        fig, axes = plt.subplots(1, len(jobs), figsize=(3.2 * len(jobs), 3.8),
                                 sharey=False)
        if len(jobs) == 1:
            axes = [axes]
        all_el = obs["el_i"].replace([np.inf, -np.inf], np.nan).dropna()
        q1, q99 = all_el.quantile(0.01), all_el.quantile(0.99)
        for ax, job, color in zip(axes, jobs, PALETTE):
            mask = (df["job_type"] == job).values
            el_j = obs["el_i"].values[mask]
            el_j = el_j[(el_j >= q1) & (el_j <= q99) & np.isfinite(el_j)]
            _kde_hist(ax, el_j, color, bins=25)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axvline(np.mean(el_j), color=color, linewidth=1.5,
                       linestyle="--", alpha=0.9)
            ax.set_title(JOB_LABELS[job], fontsize=10)
            ax.set_xlabel("$EL_i$", fontsize=9)
            ax.set_ylabel("Density" if ax == axes[0] else "", fontsize=9)
            ax.text(0.97, 0.93, f"μ = {np.mean(el_j):.2f}",
                    transform=ax.transAxes, ha="right", fontsize=8.5, color=color)
        fig.suptitle("Distribution of Elasticity  $EL_i$  by Job Type\n[1st–99th pct trimmed]",
                     fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, "19_elasticity_dist_by_job", out_dir)


def _p20_job_dstar_dist(df, obs, out_dir):
    """Per-job KDE of D* — 5-panel figure."""
    with plt.rc_context(BASE_STYLE):
        jobs = [j for j in JOB_ORDER if j in df["job_type"].unique()]
        fig, axes = plt.subplots(1, len(jobs), figsize=(3.2 * len(jobs), 3.8),
                                 sharey=False)
        if len(jobs) == 1:
            axes = [axes]
        all_ds = obs["dstar_i"].replace([np.inf, -np.inf], np.nan).dropna()
        q5, q95 = all_ds.quantile(0.05), all_ds.quantile(0.95)
        for ax, job, color in zip(axes, jobs, PALETTE):
            mask = (df["job_type"] == job).values
            ds_j = obs["dstar_i"].values[mask]
            ds_j = ds_j[(ds_j >= q5) & (ds_j <= q95) & np.isfinite(ds_j)]
            _kde_hist(ax, ds_j, color, bins=25)
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axvline(np.median(ds_j), color=color, linewidth=1.5,
                       linestyle="--", alpha=0.9)
            ax.set_title(JOB_LABELS[job], fontsize=10)
            ax.set_xlabel("$D_i^*$ (minutes)", fontsize=9)
            ax.set_ylabel("Density" if ax == axes[0] else "", fontsize=9)
            med_label = f"Med = {np.median(ds_j):.1f}"
            ax.text(0.97, 0.93, med_label,
                    transform=ax.transAxes, ha="right", fontsize=8.5, color=color)
        fig.suptitle("Distribution of Willingness-to-Take Threshold  $D_i^*$  by Job Type\n"
                     "[5th–95th pct trimmed; dashed = median]",
                     fontsize=11, fontweight="bold", y=1.02)
        fig.tight_layout()
        _save(fig, "20_dstar_dist_by_job", out_dir)


# ── Master function ───────────────────────────────────────────────────────────

def make_all_plots(panel: pd.DataFrame, dm: dict, fit: dict,
                   target_result: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    obs = target_result["obs_level"]
    df  = dm["df"].copy()
    D   = dm["D"]
    p   = obs["p_hat"].values
    se_ame = target_result["targets"].get("se_ame", 0)

    _p01_choice_by_job(df, out_dir)
    _p02_choice_by_release(df, out_dir)
    _p03_choice_by_fatigue(df, out_dir)
    _p04_choice_by_weather(df, out_dir)
    _p05_dist_freeway(df, out_dir)
    _p06_dist_coastal(df, out_dir)
    _p07_dist_delta(df, D, out_dir)
    _p08_predicted_prob_overall(df, D, p, out_dir)
    _p09_predicted_prob_by_job(df, D, p, out_dir)
    _p10_dist_slope(obs, out_dir)
    _p11_dist_intercept(obs, out_dir)
    _p12_dist_me(obs, out_dir)
    _p13_dist_elasticity(obs, out_dir)
    _p14_dist_dstar(obs, out_dir)
    _p15_job_ame(df, obs, se_ame, out_dir)
    _p16_job_elasticity(df, obs, out_dir)
    _p17_sunny_cloudy(df, obs, out_dir)
    _p18_fatigue_contrasts(df, obs, out_dir)
    _p19_job_elasticity_dist(df, obs, out_dir)
    _p20_job_dstar_dist(df, obs, out_dir)

    print(f"  20 plots saved to {out_dir}")

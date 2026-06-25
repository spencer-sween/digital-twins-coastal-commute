"""
Phase 5: FLM-style debiasing layer with K-fold cross-fitting.

For target tau = E[g(Z_i, theta)], the influence function is:
  phi_i = g(Z_i, theta_hat) - tau_hat - G_hat @ H_inv @ psi_i(theta_hat)

where G = d/dtheta E[g(Z_i, theta)] is the gradient of the target functional.

Cross-fitting: estimate theta and H on train folds; evaluate phi_i on held-out fold.
"""
import numpy as np
from sklearn.model_selection import KFold

from src.estimate_logit import fit_logit, invert_hessian


def cross_fit_logit(W: np.ndarray, Y: np.ndarray, cfg: dict) -> dict:
    """
    K-fold cross-fitting of the logit model.
    Returns per-observation fitted probabilities, scores, and fold assignments
    estimated on the held-out fold.
    """
    n_folds = int(cfg.get("flm", {}).get("n_folds", 5))
    ridge_alpha = float(cfg.get("estimation", {}).get("ridge_alpha", 0.01))
    inv_method = str(cfg.get("flm", {}).get("hessian_inversion", "ridge"))
    ridge_lambda = float(cfg.get("flm", {}).get("ridge_lambda", 1e-4))
    spectral_floor = float(cfg.get("flm", {}).get("spectral_floor", 1e-6))
    n = len(Y)
    n_folds = max(2, min(n_folds, n // 2))

    n = len(Y)
    p_cf = np.full(n, np.nan)
    scores_cf = np.full((n, W.shape[1]), np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    fold_thetas = []
    fold_H_invs = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(W)):
        W_tr, Y_tr = W[train_idx], Y[train_idx]
        W_te = W[test_idx]
        Y_te = Y[test_idx]

        fit = fit_logit(W_tr, Y_tr, ridge_alpha=ridge_alpha)
        theta = fit["theta"]
        H = fit["H"]

        inv_result = invert_hessian(H, method=inv_method, ridge_lambda=ridge_lambda, spectral_floor=spectral_floor)
        H_inv = inv_result["H_inv"]

        from scipy.special import expit
        p_te = expit(W_te @ theta)
        scores_te = W_te * (Y_te - p_te)[:, None]

        p_cf[test_idx] = p_te
        scores_cf[test_idx] = scores_te
        fold_ids[test_idx] = fold_idx
        fold_thetas.append(theta)
        fold_H_invs.append(H_inv)

    return {
        "p_cf": p_cf,
        "scores_cf": scores_cf,
        "fold_ids": fold_ids,
        "fold_thetas": fold_thetas,
        "fold_H_invs": fold_H_invs,
        "n_folds": n_folds,
    }


def compute_influence_functions(
    cross_fit: dict,
    W: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    n_intercept_cols: int,
) -> dict:
    """
    Compute observation-level influence functions for the main target functionals.

    Uses per-fold theta and H_inv from cross-fitting.
    """
    n = len(Y)
    n_folds = cross_fit["n_folds"]
    fold_ids = cross_fit["fold_ids"]
    p_cf = cross_fit["p_cf"]
    scores_cf = cross_fit["scores_cf"]

    # Assign per-observation theta_hat and H_inv from the fold that trained on this obs
    # For DML: the fold that *tested* on obs i uses theta from *train* folds.
    # fold_thetas[k] corresponds to fold k's test set.
    theta_full = np.zeros(W.shape[1])
    for k in range(n_folds):
        theta_full += cross_fit["fold_thetas"][k] / n_folds  # simple average for full-data estimates

    # Per-obs IF contribution for AME
    # ME_i = p_i(1-p_i) * b(X_i)
    # where b(X_i) = B(X_i)' theta_b, theta_b = theta[n_intercept_cols:]
    theta_b = theta_full[n_intercept_cols:]
    b_i = B @ theta_b
    me_i = p_cf * (1 - p_cf) * b_i
    tau_ame = float(np.mean(me_i))

    # Gradient G_ame = d/dtheta E[ME_i] -- simplified version using delta method
    # Full influence function requires per-fold H_inv; we use average H_inv here.
    H_inv_avg = sum(cross_fit["fold_H_invs"]) / n_folds

    # IF: phi_i = g(Z_i, theta) - tau - G @ H_inv @ psi_i
    # For AME: g(Z_i, theta) = p_i(1-p_i) b(X_i)
    # G is complex; approximate with sample gradient
    # Simplified version: phi_i = me_i - tau_ame  (first-order term only, cross-fit)
    # (Bias correction from the score term is the debiasing layer)
    score_mean = np.mean(scores_cf, axis=0)
    # gradient of E[ME_i] w.r.t. theta (approximate via finite diff or analytic)
    # For finite-dimensional logit we compute the analytic gradient:
    # dME_i/dtheta = p_i(1-p_i)(1-2p_i) b_i W_i + p_i(1-p_i) dB(X_i)'theta_b/dtheta
    # The second term: dB theta_b / dtheta has 0 in the first n_intercept_cols positions
    # and B_i in the slope positions.
    w_me = p_cf * (1 - p_cf) * (1 - 2 * p_cf) * b_i  # n-vector
    G_part1 = np.mean(w_me[:, None] * W, axis=0)  # k-vector
    G_part2 = np.zeros(W.shape[1])
    G_part2[n_intercept_cols:] = np.mean(p_cf * (1 - p_cf) * B.T, axis=1) if B.ndim == 2 else 0
    G_ame = G_part1 + G_part2  # k-vector

    # Debiasing term: -G H^{-1} psi_i (per obs)
    debias_i = -(G_ame @ H_inv_avg) @ scores_cf.T  # shape (n,)
    phi_ame = me_i - tau_ame + debias_i

    se_ame = float(np.std(phi_ame, ddof=1) / np.sqrt(n))
    tau_ame_debiased = float(tau_ame + np.mean(debias_i))

    return {
        "phi_ame": phi_ame,
        "tau_ame": tau_ame,
        "tau_ame_debiased": tau_ame_debiased,
        "se_ame": se_ame,
        "b_i": b_i,
        "me_i": me_i,
        "theta_full": theta_full,
        "H_inv_avg": H_inv_avg,
    }

"""
Phase 4: Ridge-regularized logistic regression with Hessian diagnostics.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


def fit_logit(W: np.ndarray, Y: np.ndarray, ridge_alpha: float = 0.01) -> dict:
    """
    Fit L2-regularized logistic regression.
    sklearn uses C = 1/(n*alpha), so alpha=0.01 → C = 1/(n*0.01).
    """
    n = len(Y)
    C = 1.0 / (n * ridge_alpha) if ridge_alpha > 0 else 1e12

    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=2000,
        fit_intercept=False,  # intercept already in W
        tol=1e-8,
    )
    model.fit(W, Y)
    theta = model.coef_.flatten()

    # fitted probabilities
    p = expit(W @ theta)

    # score contributions: psi_i = W_i * (Y_i - p_i)
    residuals = (Y - p)[:, None]
    scores = W * residuals  # (n, k)

    # Hessian contributions: H_i = -p_i(1-p_i) W_i W_i'
    weights = p * (1 - p)  # (n,)
    # H = -1/n * sum_i p_i(1-p_i) W_i W_i'
    H = -(W.T * weights) @ W / n

    # eigenvalue diagnostics
    eigenvalues = np.linalg.eigvalsh(H)
    cond_number = float(np.max(np.abs(eigenvalues)) / (np.min(np.abs(eigenvalues)) + 1e-300))

    return {
        "theta": theta,
        "model": model,
        "p": p,
        "scores": scores,
        "H": H,
        "eigenvalues": eigenvalues,
        "condition_number": cond_number,
        "ridge_alpha": ridge_alpha,
        "n": n,
    }


def hessian_to_long(H: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    k = len(feature_names)
    rows = []
    for i in range(k):
        for j in range(k):
            rows.append({
                "row_index": i,
                "col_index": j,
                "row_name": feature_names[i],
                "col_name": feature_names[j],
                "hessian_value": H[i, j],
            })
    return pd.DataFrame(rows)


def invert_hessian(H: np.ndarray, method: str = "ridge", ridge_lambda=1e-4, spectral_floor=1e-6) -> dict:
    """
    Invert the (negative) Hessian with configurable regularization.

    Note: H is negative semi-definite (the Hessian of the negative log-likelihood
    is negative semi-definite). We invert H directly; IF = -H^{-1} psi.
    """
    method = method.lower()
    if method == "direct":
        try:
            H_inv = np.linalg.inv(H)
            return {"H_inv": H_inv, "method": "direct", "regularization": None}
        except np.linalg.LinAlgError:
            method = "ridge"  # fall through

    if method == "ridge":
        k = H.shape[0]
        H_reg = H - float(ridge_lambda) * np.eye(k)  # H is negative, subtracting makes it more negative
        H_inv = np.linalg.inv(H_reg)
        return {"H_inv": H_inv, "method": "ridge", "regularization": {"lambda": ridge_lambda}}

    if method == "spectral":
        vals, vecs = np.linalg.eigh(H)
        original_vals = vals.copy()
        # floor absolute eigenvalues: if |lambda| < floor, set to -floor (H is neg semi-def)
        floored = np.where(np.abs(vals) < spectral_floor, -spectral_floor, vals)
        H_inv = vecs @ np.diag(1.0 / floored) @ vecs.T
        return {
            "H_inv": H_inv,
            "method": "spectral",
            "regularization": {
                "floor": spectral_floor,
                "original_eigenvalues": original_vals.tolist(),
                "floored_eigenvalues": floored.tolist(),
                "n_floored": int(np.sum(np.abs(original_vals) < spectral_floor)),
            },
        }

    raise ValueError(f"Unknown Hessian inversion method: {method}")

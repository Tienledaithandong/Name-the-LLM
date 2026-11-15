"""k-robust prediction calibration via damped Sinkhorn-Knopp balancing.

Unlike :mod:`src.crossed`'s one-per-model Hungarian assignment, this calibration
assumes **no prompt groups and no passages-per-prompt count k**.  It gently
balances the predicted class marginals of a score matrix toward a target prior,
which is safe whether the hidden test has k=2 or k=18 passages per prompt.  The
community forensic evidence (see ``docs/strategy_review.md``) shows Hard Hungarian
can *hurt* when k is small because clustering merges unrelated prompts into fake
full-factorial blocks; global marginal balancing has no such failure mode.

Label-free and transductive (uses only the batch's own scores).  Never applied
inside reported grouped-CV *training* scores.
"""

from __future__ import annotations

import numpy as np


class CalibrationError(ValueError):
    """Raised for an unusable score matrix or calibration parameter."""


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sinkhorn_calibrate(
    scores,
    *,
    strength: float = 0.30,
    target_mix: float = 0.35,
    temperature: float = 1.0,
    n_iter: int = 50,
    target=None,
) -> np.ndarray:
    """Return calibrated row-stochastic probabilities for a score matrix.

    Parameters
    ----------
    scores : (n_rows, n_classes) array of finite margins (e.g. blend z-scores).
    strength : damping in [0, 1].  0 returns the plain softmax (argmax unchanged);
        1 moves fully toward the column-balanced matrix.
    target_mix : in [0, 1].  The target column prior is
        ``(1 - target_mix) * uniform + target_mix * empirical_predicted_marginal``.
        0 balances toward uniform (1/K); 1 keeps the batch's own predicted mix
        (no net balancing).  Ignored when ``target`` is given.
    temperature : softmax temperature applied to the margins.
    n_iter : Sinkhorn column/row scaling iterations.
    target : optional explicit column prior (length K, normalised internally).

    The result is ``(1 - strength) * P + strength * Q`` where ``P`` is the softmax
    and ``Q`` is the Sinkhorn-balanced matrix whose column sums approach
    ``n_rows * target``.  Predictions are the argmax of the returned matrix.
    """
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim != 2:
        raise CalibrationError("scores must be 2-D (rows x classes)")
    if not np.isfinite(S).all():
        raise CalibrationError("calibration needs finite scores; normalise sentinels first")
    if not 0.0 <= strength <= 1.0:
        raise CalibrationError("strength must be in [0, 1]")
    if not 0.0 <= target_mix <= 1.0:
        raise CalibrationError("target_mix must be in [0, 1]")
    n, k = S.shape
    P = _softmax(S, temperature)
    if strength == 0.0 or n == 0:
        return P

    if target is None:
        empirical = P.mean(axis=0)
        uniform = np.full(k, 1.0 / k)
        prior = (1.0 - target_mix) * uniform + target_mix * empirical
    else:
        prior = np.asarray(target, dtype=np.float64)
        if prior.shape != (k,):
            raise CalibrationError("target must have one entry per class")
        prior = prior / prior.sum()
    col_target = n * prior

    Q = P.copy()
    for _ in range(int(n_iter)):
        col = Q.sum(axis=0)
        col[col < 1e-12] = 1e-12
        Q = Q * (col_target / col)
        row = Q.sum(axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        Q = Q / row
    return (1.0 - strength) * P + strength * Q


def calibrate_predict(scores, **kwargs) -> np.ndarray:
    """Predicted class index per row after damped Sinkhorn calibration."""
    return sinkhorn_calibrate(scores, **kwargs).argmax(axis=1)
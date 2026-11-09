"""Metrics for cell probability predictions.

Only cells that were still *unknown* in a snapshot are scored - predicting
already revealed cells is trivial and would inflate every metric.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def cell_metrics(probs: np.ndarray, Y: np.ndarray, unknown: np.ndarray) -> dict[str, float]:
    """Classification metrics over all unknown cells plus the "next-shot" hit rate.

    ``next_shot_hit_rate`` answers the question that matters in the game:
    if we shoot at the unknown cell with the highest predicted probability,
    how often is there a ship?
    """
    p = np.clip(probs[unknown], 1e-6, 1 - 1e-6)
    y = Y[unknown]
    masked = np.where(unknown, probs, -np.inf)
    best = masked.argmax(axis=1)
    hit_rate = Y[np.arange(len(Y)), best].mean()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "next_shot_hit_rate": float(hit_rate),
    }


def density_to_probability(density: np.ndarray, expected_cells: np.ndarray) -> np.ndarray:
    """Scale density maps so that each row sums to the expected number of ship cells."""
    totals = density.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.clip(density / totals * expected_cells[:, None], 0.0, 1.0)

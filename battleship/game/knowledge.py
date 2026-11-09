"""Facts that follow directly from the rules and the visible board.

Besides the cells that are certainly empty, this module contains the
constraint search over all ship placements that are still possible: for every
remaining ship, every position on the board is enumerated and checked against
what is already known (misses, sunk ships, the no-touch rule). Counting how
many legal placements cover each cell gives a probability density.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from battleship.game.board import CellState, Observation


def dilate(mask: np.ndarray, diagonal_only: bool = False) -> np.ndarray:
    """Cells that are (8- or diagonally-) adjacent to a ``True`` cell."""
    padded = np.pad(mask, 1)
    n_rows, n_cols = mask.shape
    out = np.zeros_like(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr, dc) == (0, 0) or (diagonal_only and (dr == 0 or dc == 0)):
                continue
            out |= padded[1 + dr : 1 + dr + n_rows, 1 + dc : 1 + dc + n_cols]
    return out


def known_empty_mask(obs: Observation) -> np.ndarray:
    """Unknown cells that certainly contain no ship.

    With the no-touch rule, all cells around a sunk ship and the diagonal
    neighbours of a hit are guaranteed to be water.
    """
    if obs.config.allow_touching:
        return np.zeros_like(obs.unknown_mask)
    sunk = obs.grid == CellState.SUNK
    hits = obs.grid == CellState.HIT
    empty = dilate(sunk) | dilate(hits, diagonal_only=True)
    return empty & obs.unknown_mask


#: How strongly placements that cover an unsunk hit are preferred.
HIT_WEIGHT = 50.0


def _placement_weights(blocked: np.ndarray, hits: np.ndarray, length: int, no_touch: bool) -> np.ndarray:
    """Weights of all horizontal placements of a ship (shape ``n x n-L+1``)."""
    n = blocked.shape[0]
    if length > n:
        return np.zeros((n, 0))
    window = (1, length)
    blocked_any = sliding_window_view(blocked, window).any(axis=(-1, -2))
    hit_count = sliding_window_view(hits, window).sum(axis=(-1, -2))
    valid = ~blocked_any
    if no_touch:
        # A hit directly next to the ship would belong to a touching ship.
        hp = np.pad(hits, 1)
        left = hp[1 : n + 1, 0 : n - length + 1]
        right = hp[1 : n + 1, length + 1 : n + 2]
        above = sliding_window_view(hp[0:n, 1 : n + 1], window).any(axis=(-1, -2))
        below = sliding_window_view(hp[2 : n + 2, 1 : n + 1], window).any(axis=(-1, -2))
        valid &= ~(left | right | above | below)
    return valid * np.power(HIT_WEIGHT, hit_count)


def placement_density(obs: Observation) -> np.ndarray:
    """Weighted number of legal ship placements covering each unknown cell."""
    grid = obs.grid
    n = obs.config.size
    no_touch = not obs.config.allow_touching
    hits = grid == CellState.HIT
    blocked = (grid == CellState.MISS) | (grid == CellState.SUNK) | known_empty_mask(obs)

    density = np.zeros((n, n), dtype=float)
    for length, count in Counter(obs.remaining).items():
        for transpose in (False, True):
            b = blocked.T if transpose else blocked
            h = hits.T if transpose else hits
            weights = _placement_weights(b, h, length, no_touch) * count
            cover = np.zeros((n, n), dtype=float)
            for offset in range(length):
                cover[:, offset : offset + n - length + 1] += weights
            density += cover.T if transpose else cover

    density[~obs.unknown_mask] = 0.0
    return density

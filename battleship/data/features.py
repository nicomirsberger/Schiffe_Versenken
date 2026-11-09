"""Turn an :class:`Observation` into a numeric feature vector.

Layout (for a 10x10 board and the default fleet):

* 4 x 100 one-hot values - state of every cell (unknown, miss, hit, sunk)
* 100 values            - cell is known to be empty (no-touch rule)
* 100 values            - placement density of the constraint search, scaled to [0, 1]
                          (engineered feature, added in CRISP-DM iteration 2)
* max_ship_length values - fraction of ships of each length still afloat
"""

from __future__ import annotations

import numpy as np

from battleship.config import GameConfig
from battleship.game.board import CellState, Observation
from battleship.game.knowledge import known_empty_mask, placement_density

FEATURE_VERSION = 2
N_STATES = len(CellState)


def feature_size(config: GameConfig) -> int:
    cells = config.size * config.size
    return cells * (N_STATES + 2) + config.max_ship_length


def density_feature(features: np.ndarray, config: GameConfig) -> np.ndarray:
    """Extract the (scaled) density part of encoded feature vectors."""
    cells = config.size * config.size
    return features[..., cells * (N_STATES + 1) : cells * (N_STATES + 2)]


def encode_observation(obs: Observation) -> np.ndarray:
    """Encode one observation as a ``float32`` vector of length :func:`feature_size`."""
    config = obs.config
    one_hot = np.eye(N_STATES, dtype=np.float32)[obs.grid.astype(int)]  # (n, n, 4)
    cell_states = one_hot.transpose(2, 0, 1).reshape(-1)
    empty = known_empty_mask(obs).astype(np.float32).reshape(-1)
    density = placement_density(obs)
    if density.max() > 0:
        density = density / density.max()
    density = density.astype(np.float32).reshape(-1)

    remaining = np.zeros(config.max_ship_length, dtype=np.float32)
    for length in range(1, config.max_ship_length + 1):
        total = config.fleet_count(length)
        if total:
            remaining[length - 1] = obs.remaining.count(length) / total
    return np.concatenate([cell_states, empty, density, remaining])


def encode_many(observations: list[Observation]) -> np.ndarray:
    return np.stack([encode_observation(o) for o in observations])


def decode_observation(features: np.ndarray, config: GameConfig) -> Observation:
    """Inverse of :func:`encode_observation` (used for baselines in evaluation)."""
    n = config.size
    cells = n * n
    grid = features[: cells * N_STATES].reshape(N_STATES, n, n).argmax(axis=0).astype(np.int8)
    fractions = features[cells * (N_STATES + 2) :]
    remaining: list[int] = []
    for length in range(1, config.max_ship_length + 1):
        remaining += [length] * int(round(fractions[length - 1] * config.fleet_count(length)))
    return Observation(grid, tuple(sorted(remaining, reverse=True)), config)

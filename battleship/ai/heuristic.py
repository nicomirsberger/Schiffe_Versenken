"""Simple rule-based agents used as baselines and as data generators."""

from __future__ import annotations

import numpy as np

from battleship.ai.base import Agent
from battleship.game.board import CellState, Coord, Observation
from battleship.game.knowledge import known_empty_mask


class RandomAgent(Agent):
    """Shoots at a uniformly random unknown cell."""

    name = "random"

    def choose_shot(self, obs: Observation, rng: np.random.Generator) -> Coord:
        allowed = obs.unknown_mask
        self.last_scores = allowed.astype(float)
        return self.pick_best(np.ones(allowed.shape), allowed, rng)


class HuntTargetAgent(Agent):
    """Classic human strategy.

    * Hunt: shoot on a checkerboard pattern (every ship has length >= 2).
    * Target: after a hit, shoot its neighbours - along the line if the
      direction of the ship is already known.
    """

    name = "hunt_target"

    def choose_shot(self, obs: Observation, rng: np.random.Generator) -> Coord:
        n = obs.config.size
        allowed = obs.unknown_mask & ~known_empty_mask(obs)
        if not allowed.any():
            allowed = obs.unknown_mask
        hits = obs.grid == CellState.HIT
        scores = np.zeros((n, n))

        for r, c in np.argwhere(hits):
            horizontal_line = (c > 0 and hits[r, c - 1]) or (c < n - 1 and hits[r, c + 1])
            vertical_line = (r > 0 and hits[r - 1, c]) or (r < n - 1 and hits[r + 1, c])
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    along = (dr == 0 and horizontal_line) or (dc == 0 and vertical_line)
                    scores[rr, cc] = max(scores[rr, cc], 2.0 if along else 1.0)

        if not (scores * allowed).any():
            rows, cols = np.indices((n, n))
            scores = ((rows + cols) % 2 == 0).astype(float)
            if not (scores * allowed).any():
                scores = np.ones((n, n))

        self.last_scores = scores * allowed
        return self.pick_best(scores, allowed, rng)

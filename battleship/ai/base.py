"""Common interface of all agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from battleship.game.board import Coord, Observation


class Agent(ABC):
    """An attacker that chooses the next cell to shoot at.

    Agents only see an :class:`Observation`, never the hidden ship positions.
    ``last_scores`` holds the score map of the latest decision so the UI can
    visualise what the AI is "thinking".
    """

    name: str = "agent"

    def __init__(self) -> None:
        self.last_scores: np.ndarray | None = None

    @abstractmethod
    def choose_shot(self, obs: Observation, rng: np.random.Generator) -> Coord:
        """Return the coordinate of the next shot (must be an unknown cell)."""

    @staticmethod
    def pick_best(scores: np.ndarray, allowed: np.ndarray, rng: np.random.Generator) -> Coord:
        """Pick the highest-scoring allowed cell (random tie-break).

        Falls back to a random allowed cell if no cell has a positive score.
        """
        masked = np.where(allowed, scores, -np.inf)
        best = masked.max()
        if not np.isfinite(best) or best <= 0:
            candidates = np.argwhere(allowed)
        else:
            candidates = np.argwhere(np.isclose(masked, best))
        if len(candidates) == 0:
            raise RuntimeError("No cell left to shoot at.")
        row, col = candidates[rng.integers(len(candidates))]
        return int(row), int(col)

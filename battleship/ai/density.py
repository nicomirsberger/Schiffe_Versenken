"""Agent based purely on the constraint search (no learning).

It is the "search" component of the strong AI and a strong baseline.
"""

from __future__ import annotations

import numpy as np

from battleship.ai.base import Agent
from battleship.game.board import Coord, Observation
from battleship.game.knowledge import known_empty_mask, placement_density


class DensityAgent(Agent):
    """Shoots at the cell covered by the most legal ship placements."""

    name = "density"

    def choose_shot(self, obs: Observation, rng: np.random.Generator) -> Coord:
        density = placement_density(obs)
        self.last_scores = density / density.max() if density.max() > 0 else density
        allowed = obs.unknown_mask & ~known_empty_mask(obs)
        if not allowed.any():
            allowed = obs.unknown_mask
        return self.pick_best(density, allowed, rng)

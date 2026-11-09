"""The neural network opponent."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from battleship.ai.base import Agent
from battleship.ai.density import placement_density
from battleship.ai.model import DEFAULT_MODEL_PATH, CellProbabilityModel
from battleship.game.board import Coord, Observation
from battleship.game.knowledge import known_empty_mask


class NeuralAgent(Agent):
    """Shoots where the neural network expects a ship.

    With ``use_search=True`` the network's probabilities are combined with
    the constraint search (:func:`placement_density`): impossible cells are
    pruned and the remaining cells are ranked by the product of both
    estimates ("product of experts").
    """

    def __init__(self, model: CellProbabilityModel, use_search: bool = True) -> None:
        super().__init__()
        self.model = model
        self.use_search = use_search
        self.name = "neural_search" if use_search else "neural"

    @classmethod
    def from_file(cls, path: Path = DEFAULT_MODEL_PATH, use_search: bool = True) -> NeuralAgent:
        return cls(CellProbabilityModel.load(path), use_search=use_search)

    def score(self, obs: Observation) -> np.ndarray:
        if obs.config != self.model.config:
            raise ValueError("The model was trained for a different game configuration.")
        scores = self.model.predict_observation(obs)
        if self.use_search:
            density = placement_density(obs)
            if density.max() > 0:
                scores = scores * (density / density.max())
        allowed = obs.unknown_mask & ~known_empty_mask(obs)
        return np.where(allowed, scores, 0.0)

    def choose_shot(self, obs: Observation, rng: np.random.Generator) -> Coord:
        scores = self.score(obs)
        self.last_scores = scores / scores.max() if scores.max() > 0 else scores
        allowed = obs.unknown_mask & ~known_empty_mask(obs)
        if not allowed.any():
            allowed = obs.unknown_mask
        return self.pick_best(scores, allowed, rng)

"""Difficulty levels - the only place that maps a level to an agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from battleship.ai.base import Agent
from battleship.ai.density import DensityAgent
from battleship.ai.heuristic import HuntTargetAgent, RandomAgent
from battleship.ai.model import DEFAULT_MODEL_PATH


@dataclass(frozen=True)
class Difficulty:
    key: str
    label: str
    description: str


DIFFICULTIES: tuple[Difficulty, ...] = (
    Difficulty("beginner", "Beginner", "Shoots at random cells."),
    Difficulty("easy", "Easy", "Hunt & target: checkerboard search, then attacks around hits."),
    Difficulty("medium", "Medium", "Neural network predicts where your ships are."),
    Difficulty("hard", "Hard", "Neural network + constraint search over all possible fleets."),
)


def create_agent(level: str, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[Agent, str | None]:
    """Create the agent for a difficulty level.

    Returns the agent and an optional warning (e.g. when the trained model is
    missing and a non-learning fallback is used).
    """
    if level == "beginner":
        return RandomAgent(), None
    if level == "easy":
        return HuntTargetAgent(), None
    if level in ("medium", "hard"):
        try:
            from battleship.ai.neural import NeuralAgent

            return NeuralAgent.from_file(model_path, use_search=(level == "hard")), None
        except Exception as error:  # any loading problem -> playable fallback
            return DensityAgent(), f"Model unavailable ({error}); using the search agent instead."
    raise ValueError(f"Unknown difficulty: {level}")

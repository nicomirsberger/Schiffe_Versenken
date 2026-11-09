"""AI opponents (agents) that decide where to shoot."""

from battleship.ai.base import Agent
from battleship.ai.density import DensityAgent, placement_density
from battleship.ai.heuristic import HuntTargetAgent, RandomAgent
from battleship.ai.neural import NeuralAgent

__all__ = [
    "Agent",
    "DensityAgent",
    "HuntTargetAgent",
    "NeuralAgent",
    "RandomAgent",
    "placement_density",
]

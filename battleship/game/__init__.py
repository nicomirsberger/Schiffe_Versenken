"""Game rules - independent of any AI or user interface."""

from battleship.game.board import (
    Board,
    CellState,
    Coord,
    Observation,
    Ship,
    ShotOutcome,
    ShotResult,
)
from battleship.game.match import Game, Player, simulate_attack
from battleship.game.placement import can_place, random_fleet, ship_cells

__all__ = [
    "Board",
    "CellState",
    "Coord",
    "Game",
    "Observation",
    "Player",
    "Ship",
    "ShotOutcome",
    "ShotResult",
    "can_place",
    "random_fleet",
    "ship_cells",
    "simulate_attack",
]

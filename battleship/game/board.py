"""Board, ships and shot logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

from battleship.config import DEFAULT_CONFIG, GameConfig

Coord = tuple[int, int]


class CellState(IntEnum):
    """What the attacker knows about a cell of the opponent's board."""

    UNKNOWN = 0
    MISS = 1
    HIT = 2  # hit, but the ship is not sunk yet
    SUNK = 3


class ShotResult(Enum):
    MISS = "miss"
    HIT = "hit"
    SUNK = "sunk"


@dataclass(frozen=True)
class Ship:
    """A ship occupying a straight line of cells."""

    cells: tuple[Coord, ...]

    @property
    def length(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class ShotOutcome:
    coord: Coord
    result: ShotResult
    ship: Ship | None = None  # set when a ship was sunk


@dataclass(frozen=True)
class Observation:
    """Everything an attacker is allowed to know about the opponent's board.

    Attributes:
        grid: ``size x size`` array of :class:`CellState` values.
        remaining: Lengths of the ships that are not sunk yet (longest first).
        config: The rules of the game.
    """

    grid: np.ndarray
    remaining: tuple[int, ...]
    config: GameConfig

    @property
    def unknown_mask(self) -> np.ndarray:
        return self.grid == CellState.UNKNOWN

    @property
    def shots_fired(self) -> int:
        return int(np.count_nonzero(self.grid != CellState.UNKNOWN))


class Board:
    """The hidden fleet of one player plus the shots fired at it."""

    def __init__(self, ships: list[Ship], config: GameConfig = DEFAULT_CONFIG) -> None:
        # Imported here to avoid a circular import (placement uses Ship).
        from battleship.game.placement import is_valid_fleet

        if not is_valid_fleet(ships, config):
            raise ValueError("Invalid fleet placement for the given configuration.")
        self.config = config
        self.ships: list[Ship] = list(ships)
        self._cell_to_ship: dict[Coord, int] = {
            cell: index for index, ship in enumerate(self.ships) for cell in ship.cells
        }
        self._hits: list[set[Coord]] = [set() for _ in self.ships]
        self._grid = np.full((config.size, config.size), CellState.UNKNOWN, dtype=np.int8)

    # ------------------------------------------------------------------ queries
    def in_bounds(self, coord: Coord) -> bool:
        row, col = coord
        return 0 <= row < self.config.size and 0 <= col < self.config.size

    def is_shot(self, coord: Coord) -> bool:
        return bool(self._grid[coord] != CellState.UNKNOWN)

    def is_sunk(self, ship_index: int) -> bool:
        return len(self._hits[ship_index]) == self.ships[ship_index].length

    @property
    def all_sunk(self) -> bool:
        return all(self.is_sunk(i) for i in range(len(self.ships)))

    @property
    def shots_fired(self) -> int:
        return int(np.count_nonzero(self._grid != CellState.UNKNOWN))

    def remaining_ship_lengths(self) -> tuple[int, ...]:
        lengths = [s.length for i, s in enumerate(self.ships) if not self.is_sunk(i)]
        return tuple(sorted(lengths, reverse=True))

    def occupancy(self) -> np.ndarray:
        """Boolean ``size x size`` array that is ``True`` where a ship lies."""
        occ = np.zeros((self.config.size, self.config.size), dtype=bool)
        for cell in self._cell_to_ship:
            occ[cell] = True
        return occ

    def observation(self) -> Observation:
        """The opponent's (attacker's) view of this board."""
        return Observation(self._grid.copy(), self.remaining_ship_lengths(), self.config)

    # ------------------------------------------------------------------ actions
    def fire(self, coord: Coord) -> ShotOutcome:
        """Fire at ``coord`` and return what happened."""
        coord = (int(coord[0]), int(coord[1]))
        if not self.in_bounds(coord):
            raise ValueError(f"Shot {coord} is outside the board.")
        if self.is_shot(coord):
            raise ValueError(f"Cell {coord} has already been shot.")

        ship_index = self._cell_to_ship.get(coord)
        if ship_index is None:
            self._grid[coord] = CellState.MISS
            return ShotOutcome(coord, ShotResult.MISS)

        self._hits[ship_index].add(coord)
        ship = self.ships[ship_index]
        if self.is_sunk(ship_index):
            for cell in ship.cells:
                self._grid[cell] = CellState.SUNK
            return ShotOutcome(coord, ShotResult.SUNK, ship)

        self._grid[coord] = CellState.HIT
        return ShotOutcome(coord, ShotResult.HIT)

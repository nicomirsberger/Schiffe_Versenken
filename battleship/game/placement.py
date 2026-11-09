"""Ship placement rules and random fleet generation."""

from __future__ import annotations

import numpy as np

from battleship.config import DEFAULT_CONFIG, GameConfig
from battleship.game.board import Coord, Ship

_NEIGHBOURS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def ship_cells(origin: Coord, length: int, horizontal: bool) -> tuple[Coord, ...]:
    """Cells of a ship that starts at ``origin`` (top/left end)."""
    row, col = origin
    if horizontal:
        return tuple((row, col + i) for i in range(length))
    return tuple((row + i, col) for i in range(length))


def _in_bounds(cell: Coord, size: int) -> bool:
    return 0 <= cell[0] < size and 0 <= cell[1] < size


def can_place(existing: list[Ship], new_ship: Ship, config: GameConfig = DEFAULT_CONFIG) -> bool:
    """Whether ``new_ship`` can be added to ``existing`` without breaking a rule."""
    if not all(_in_bounds(c, config.size) for c in new_ship.cells):
        return False
    occupied = {cell for ship in existing for cell in ship.cells}
    for row, col in new_ship.cells:
        if (row, col) in occupied:
            return False
        touches = any((row + dr, col + dc) in occupied for dr, dc in _NEIGHBOURS)
        if touches and not config.allow_touching:
            return False
    return True


def is_valid_fleet(ships: list[Ship], config: GameConfig = DEFAULT_CONFIG) -> bool:
    """Whether ``ships`` is a complete, rule-conforming fleet."""
    if sorted((s.length for s in ships), reverse=True) != list(config.fleet):
        return False
    for ship in ships:
        rows = {r for r, _ in ship.cells}
        cols = {c for _, c in ship.cells}
        straight = (len(rows) == 1 and sorted(cols) == list(range(min(cols), min(cols) + ship.length))) or (
            len(cols) == 1 and sorted(rows) == list(range(min(rows), min(rows) + ship.length))
        )
        if not straight:
            return False
    placed: list[Ship] = []
    for ship in ships:
        if not can_place(placed, ship, config):
            return False
        placed.append(ship)
    return True


def random_fleet(
    config: GameConfig = DEFAULT_CONFIG,
    rng: np.random.Generator | None = None,
    max_attempts: int = 1000,
) -> list[Ship]:
    """Place the whole fleet uniformly at random (longest ships first).

    If the board runs out of space the placement restarts from scratch.
    """
    rng = rng if rng is not None else np.random.default_rng()
    for _ in range(max_attempts):
        ships: list[Ship] = []
        for length in config.fleet:
            candidates = []
            for horizontal in (True, False):
                max_row = config.size - (1 if horizontal else length)
                max_col = config.size - (length if horizontal else 1)
                for row in range(max_row + 1):
                    for col in range(max_col + 1):
                        ship = Ship(ship_cells((row, col), length, horizontal))
                        if can_place(ships, ship, config):
                            candidates.append(ship)
            if not candidates:
                break
            ships.append(candidates[rng.integers(len(candidates))])
        else:
            return ships
    raise RuntimeError("Could not place the fleet - is the board too small?")

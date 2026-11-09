"""Central game configuration.

All modules receive a :class:`GameConfig` instead of hard-coding rules, so the
board size, the fleet and the rule variant can be changed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Classic fleet:
#: 1x battleship (5), 2x cruiser (4), 3x destroyer (3), 4x submarine (2).
DEFAULT_FLEET: tuple[int, ...] = (5, 4, 4, 3, 3, 3, 2, 2, 2, 2)

SHIP_NAMES: dict[int, str] = {
    5: "Battleship",
    4: "Cruiser",
    3: "Destroyer",
    2: "Submarine",
}


@dataclass(frozen=True)
class GameConfig:
    """Rules of a Battleship game.

    Attributes:
        size: Width and height of the square board.
        fleet: Lengths of all ships each player owns.
        allow_touching: Whether ships may touch each other (also diagonally).
        extra_shot_on_hit: Whether a player may shoot again after a hit.
    """

    size: int = 10
    fleet: tuple[int, ...] = DEFAULT_FLEET
    allow_touching: bool = False
    extra_shot_on_hit: bool = True

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("Board size must be at least 2.")
        if not self.fleet:
            raise ValueError("The fleet must contain at least one ship.")
        if any(length < 1 or length > self.size for length in self.fleet):
            raise ValueError("Every ship must fit on the board.")
        # Keep the fleet sorted (longest first) for deterministic behaviour.
        object.__setattr__(self, "fleet", tuple(sorted(self.fleet, reverse=True)))

    @property
    def max_ship_length(self) -> int:
        return max(self.fleet)

    @property
    def total_ship_cells(self) -> int:
        return sum(self.fleet)

    def fleet_count(self, length: int) -> int:
        """Number of ships with the given length in the fleet."""
        return self.fleet.count(length)


DEFAULT_CONFIG = GameConfig()

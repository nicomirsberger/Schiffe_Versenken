import numpy as np
import pytest

from battleship.config import DEFAULT_CONFIG, GameConfig
from battleship.game import Ship, can_place, random_fleet, ship_cells
from battleship.game.placement import is_valid_fleet


def test_ship_cells():
    assert ship_cells((1, 2), 3, horizontal=True) == ((1, 2), (1, 3), (1, 4))
    assert ship_cells((1, 2), 2, horizontal=False) == ((1, 2), (2, 2))


def test_no_touch_rule():
    existing = [Ship(((0, 0), (0, 1)))]
    diagonal = Ship(((1, 2), (2, 2)))
    far = Ship(((3, 3), (3, 4)))
    assert not can_place(existing, diagonal, DEFAULT_CONFIG)
    assert can_place(existing, far, DEFAULT_CONFIG)
    touching_allowed = GameConfig(allow_touching=True)
    assert can_place(existing, diagonal, touching_allowed)


def test_out_of_bounds():
    assert not can_place([], Ship(ship_cells((0, 8), 3, True)), DEFAULT_CONFIG)


@pytest.mark.parametrize("seed", range(20))
def test_random_fleet_is_valid(seed):
    fleet = random_fleet(DEFAULT_CONFIG, np.random.default_rng(seed))
    assert is_valid_fleet(fleet, DEFAULT_CONFIG)


def test_bent_ship_is_invalid():
    config = GameConfig(size=5, fleet=(3,))
    assert not is_valid_fleet([Ship(((0, 0), (0, 1), (1, 1)))], config)


def test_invalid_config():
    with pytest.raises(ValueError):
        GameConfig(size=4, fleet=(5,))

import numpy as np

from battleship.config import DEFAULT_CONFIG, GameConfig
from battleship.game import Board, CellState, Ship, random_fleet
from battleship.game.knowledge import known_empty_mask, placement_density


def test_density_is_zero_on_shot_cells(rng):
    board = Board(random_fleet(DEFAULT_CONFIG, rng))
    for cell in [(0, 0), (5, 5), (9, 9)]:
        board.fire(cell)
    density = placement_density(board.observation())
    assert density[0, 0] == density[5, 5] == density[9, 9] == 0
    assert density.max() > 0


def test_density_prefers_cells_next_to_hit():
    config = GameConfig(size=5, fleet=(2,))
    board = Board([Ship(((2, 2), (2, 3)))], config)
    board.fire((2, 2))
    density = placement_density(board.observation())
    best = np.unravel_index(density.argmax(), density.shape)
    assert best in {(1, 2), (3, 2), (2, 1), (2, 3)}


def test_known_empty_around_sunk_ship():
    config = GameConfig(size=5, fleet=(2,))
    board = Board([Ship(((2, 2), (2, 3)))], config)
    board.fire((2, 2))
    board.fire((2, 3))
    empty = known_empty_mask(board.observation())
    assert empty[1, 1] and empty[3, 4] and empty[2, 1]
    assert not empty[0, 0]


def test_diagonal_of_hit_is_empty():
    config = GameConfig(size=5, fleet=(3,))
    board = Board([Ship(((2, 1), (2, 2), (2, 3)))], config)
    board.fire((2, 2))
    obs = board.observation()
    assert obs.grid[2, 2] == CellState.HIT
    empty = known_empty_mask(obs)
    assert empty[1, 1] and empty[3, 3] and not empty[1, 2]

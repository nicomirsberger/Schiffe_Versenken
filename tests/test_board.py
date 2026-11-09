import pytest

from battleship.config import DEFAULT_CONFIG, GameConfig
from battleship.game import Board, CellState, Ship, ShotResult, random_fleet


def make_board() -> Board:
    config = GameConfig(size=5, fleet=(3, 2))
    return Board([Ship(((0, 0), (0, 1), (0, 2))), Ship(((2, 4), (3, 4)))], config)


def test_miss_hit_and_sunk():
    board = make_board()
    assert board.fire((4, 0)).result is ShotResult.MISS
    assert board.fire((0, 0)).result is ShotResult.HIT
    assert board.fire((0, 1)).result is ShotResult.HIT
    outcome = board.fire((0, 2))
    assert outcome.result is ShotResult.SUNK
    assert outcome.ship is not None and outcome.ship.length == 3
    grid = board.observation().grid
    assert grid[0, 0] == grid[0, 1] == grid[0, 2] == CellState.SUNK
    assert grid[4, 0] == CellState.MISS


def test_cannot_shoot_twice_or_outside():
    board = make_board()
    board.fire((1, 1))
    with pytest.raises(ValueError):
        board.fire((1, 1))
    with pytest.raises(ValueError):
        board.fire((5, 0))


def test_all_sunk_and_remaining():
    board = make_board()
    assert board.remaining_ship_lengths() == (3, 2)
    for cell in [(2, 4), (3, 4)]:
        board.fire(cell)
    assert board.remaining_ship_lengths() == (3,)
    assert not board.all_sunk
    for cell in [(0, 0), (0, 1), (0, 2)]:
        board.fire(cell)
    assert board.all_sunk


def test_observation_hides_ships():
    board = Board(random_fleet(DEFAULT_CONFIG))
    obs = board.observation()
    assert (obs.grid == CellState.UNKNOWN).all()
    assert obs.remaining == DEFAULT_CONFIG.fleet


def test_invalid_fleet_rejected():
    config = GameConfig(size=5, fleet=(3, 2))
    with pytest.raises(ValueError):
        Board([Ship(((0, 0), (0, 1), (0, 2))), Ship(((1, 0), (1, 1)))], config)  # touching

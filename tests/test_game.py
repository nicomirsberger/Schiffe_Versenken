import pytest

from battleship.ai import DensityAgent
from battleship.config import GameConfig
from battleship.game import Board, Game, Player, Ship, ShotResult, random_fleet


def test_turns_and_extra_shot(small_config, rng):
    human = Board(random_fleet(small_config, rng), small_config)
    ai_ships = [Ship(((0, 0), (0, 1), (0, 2))), Ship(((2, 0), (3, 0))), Ship(((5, 4), (5, 5)))]
    game = Game(human, Board(ai_ships, small_config), DensityAgent(), rng)
    assert game.current is Player.HUMAN
    assert game.human_fire((0, 0)).result is ShotResult.HIT
    assert game.current is Player.HUMAN  # hit -> shoot again
    assert game.human_fire((4, 4)).result is ShotResult.MISS
    assert game.current is Player.AI
    with pytest.raises(RuntimeError):
        game.human_fire((4, 5))


def test_full_game_has_a_winner(small_config, rng):
    game = Game(
        Board(random_fleet(small_config, rng), small_config),
        Board(random_fleet(small_config, rng), small_config),
        DensityAgent(),
        rng,
        starting_player=Player.AI,
    )
    while not game.is_over:
        if game.current is Player.AI:
            game.ai_fire()
        else:
            # Human always aims at the first unknown cell.
            obs = game.ai_board.observation()
            row, col = map(int, divmod(int(obs.unknown_mask.argmax()), small_config.size))
            game.human_fire((row, col))
    assert game.winner in (Player.HUMAN, Player.AI)


def test_no_extra_shot_variant(rng):
    config = GameConfig(size=5, fleet=(2,), extra_shot_on_hit=False)
    game = Game(
        Board([Ship(((0, 0), (0, 1)))], config), Board([Ship(((0, 0), (0, 1)))], config), DensityAgent(), rng
    )
    game.human_fire((0, 0))
    assert game.current is Player.AI

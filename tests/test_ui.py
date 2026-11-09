"""Smoke tests of the pygame UI with a dummy video driver."""

import numpy as np
import pygame
import pytest

from battleship.game.match import Player
from battleship.ui.app import App, Screen
from battleship.ui.nations import NATIONS, flag_surface
from battleship.ui.ship_art import ShipScene, render_ship


@pytest.fixture
def app():
    application = App(seed=0)
    yield application
    pygame.quit()


def test_menu_setup_battle_flow(app):
    app.draw()
    app.handle_key(pygame.K_2)  # Easy
    assert app.screen is Screen.SETUP
    app.handle_key(pygame.K_SPACE)  # random fleet
    assert not app.ships_to_place
    app.handle_key(pygame.K_RETURN)
    assert app.screen is Screen.BATTLE
    app.show_heatmap = True
    app.update(0.1)
    app.draw()
    app.handle_key(pygame.K_ESCAPE)
    assert app.screen is Screen.MENU


def test_nation_cannot_fight_itself(app):
    app.player_select.handle_click(app.player_select.rect.center)  # open the list
    assert app.player_select.open
    usa_index = [n.key for n in NATIONS].index("usa")
    app.handle_menu_click(app.player_select.option_rect(usa_index).center)  # USA is the opponent
    assert app.player_select.selected == "germany"
    assert app.player_select.open  # disabled option: the list stays open

    japan_index = [n.key for n in NATIONS].index("japan")
    app.handle_menu_click(app.player_select.option_rect(japan_index).center)
    assert app.player_select.selected == "japan"
    assert app.enemy_select.disabled == {"japan"}
    assert app.player_scene.nation.key == "japan"


def test_shot_log_shows_latest_shots_first(app):
    app.start_setup(0)
    app.handle_key(pygame.K_SPACE)
    app.start_battle()
    game = app.game
    rng = np.random.default_rng(0)
    while len([1 for p, _ in game.history if p is Player.HUMAN]) < 5 and not game.is_over:
        if game.current is Player.HUMAN:
            cells = np.argwhere(game.ai_board.observation().unknown_mask)
            game.human_fire(tuple(int(v) for v in cells[rng.integers(len(cells))]))
        else:
            game.ai_fire()
    shots = app.recent_shots(Player.HUMAN)
    human = [o for p, o in game.history if p is Player.HUMAN]
    assert len(shots) == 3
    assert shots[0] == human[-1]
    app.draw()


@pytest.mark.parametrize("nation", NATIONS, ids=lambda n: n.key)
def test_every_nation_can_be_drawn_and_sunk(nation):
    pygame.init()
    surface = pygame.Surface((500, 200))
    fonts = {"small": pygame.font.SysFont("arial", 14)}
    assert render_ship(nation.ship).surface.get_width() > 0
    assert flag_surface(nation, (30, 19)).get_size() == (30, 19)
    scene = ShipScene(nation, pygame.Rect(10, 10, 420, 128), facing_right=False)
    for _ in range(30):
        scene.update(0.2, damage=0.6, defeated=True)
        scene.draw(surface, fonts)
    assert scene.sink == 1.0
    assert scene.fire_points() == []

"""Smoke test of the pygame UI with a dummy video driver."""

import pygame
import pytest

from battleship.ui.app import App, Screen


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
    app.draw()
    app.handle_key(pygame.K_ESCAPE)
    assert app.screen is Screen.MENU

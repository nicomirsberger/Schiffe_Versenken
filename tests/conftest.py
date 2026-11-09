import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from battleship.config import GameConfig  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


@pytest.fixture
def small_config() -> GameConfig:
    """A small board that keeps tests fast."""
    return GameConfig(size=6, fleet=(3, 2, 2))

"""Playable nations: name, flag and battleship class."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class Turret:
    x: float  # position along the hull (0 = stern, 1 = bow)
    facing: int  # +1 = towards the bow, -1 = towards the stern
    raised: bool = False  # superfiring turret on a barbette
    width: int = 26


@dataclass(frozen=True)
class Funnel:
    x: float
    width: int
    height: int
    rake: int = 4  # how far the top leans towards the stern


@dataclass(frozen=True)
class ShipProfile:
    """Simplified side view of a battleship class."""

    class_name: str
    turrets: tuple[Turret, ...]
    funnels: tuple[Funnel, ...]
    decks: tuple[tuple[float, float, int], ...]  # (start, end, height) superstructure blocks
    tower: tuple[float, int, int]  # (position, width, height) of the bridge tower
    mast: tuple[float, int]  # (position, height)


@dataclass(frozen=True)
class Nation:
    key: str
    name: str
    ship: ShipProfile
    draw_flag: Callable[[pygame.Surface], None]


# ----------------------------------------------------------------------- flags
def _flag_germany(s: pygame.Surface) -> None:
    w, h = s.get_size()
    for i, color in enumerate([(0, 0, 0), (221, 0, 0), (255, 206, 0)]):
        pygame.draw.rect(s, color, (0, round(i * h / 3), w, round(h / 3) + 1))


def _flag_usa(s: pygame.Surface) -> None:
    w, h = s.get_size()
    for i in range(13):
        color = (178, 34, 52) if i % 2 == 0 else (255, 255, 255)
        pygame.draw.rect(s, color, (0, round(i * h / 13), w, round(h / 13) + 1))
    cw, ch = round(w * 0.4), round(h * 7 / 13)
    pygame.draw.rect(s, (60, 59, 110), (0, 0, cw, ch))
    for row in range(3):
        for col in range(4):
            x = round((col + 0.5) * cw / 4)
            y = round((row + 0.5) * ch / 3)
            pygame.draw.circle(s, (255, 255, 255), (x, y), max(1, h // 18))


def _flag_japan(s: pygame.Surface) -> None:
    w, h = s.get_size()
    s.fill((255, 255, 255))
    pygame.draw.circle(s, (188, 0, 45), (w // 2, h // 2), round(h * 0.3))


def _flag_uk(s: pygame.Surface) -> None:
    w, h = s.get_size()
    s.fill((1, 33, 105))
    white, red = (255, 255, 255), (200, 16, 46)
    t = max(2, h // 5)
    pygame.draw.line(s, white, (0, 0), (w, h), t)
    pygame.draw.line(s, white, (0, h), (w, 0), t)
    pygame.draw.line(s, red, (0, 0), (w, h), max(1, t // 2))
    pygame.draw.line(s, red, (0, h), (w, 0), max(1, t // 2))
    pygame.draw.rect(s, white, (0, h // 2 - t, w, 2 * t))
    pygame.draw.rect(s, white, (w // 2 - t, 0, 2 * t, h))
    pygame.draw.rect(s, red, (0, h // 2 - t // 2, w, t))
    pygame.draw.rect(s, red, (w // 2 - t // 2, 0, t, h))


def _flag_france(s: pygame.Surface) -> None:
    w, h = s.get_size()
    for i, color in enumerate([(0, 35, 149), (255, 255, 255), (237, 41, 57)]):
        pygame.draw.rect(s, color, (round(i * w / 3), 0, round(w / 3) + 1, h))


# ------------------------------------------------------------------ ship classes
BISMARCK = ShipProfile(
    "Bismarck class",
    turrets=(Turret(0.17, 1), Turret(0.26, 1, True), Turret(0.75, -1, True), Turret(0.84, -1)),
    funnels=(Funnel(0.53, 18, 22),),
    decks=((0.33, 0.67, 12), (0.36, 0.49, 24)),
    tower=(0.41, 16, 40),
    mast=(0.61, 50),
)
IOWA = ShipProfile(
    "Iowa class",
    turrets=(Turret(0.16, 1, width=30), Turret(0.26, 1, True, 30), Turret(0.80, -1, width=30)),
    funnels=(Funnel(0.50, 14, 26), Funnel(0.60, 14, 24)),
    decks=((0.34, 0.67, 12), (0.37, 0.50, 24)),
    tower=(0.42, 14, 46),
    mast=(0.55, 58),
)
YAMATO = ShipProfile(
    "Yamato class",
    turrets=(Turret(0.18, 1, width=32), Turret(0.28, 1, True, 32), Turret(0.80, -1, width=32)),
    funnels=(Funnel(0.57, 22, 22, rake=9),),
    decks=((0.36, 0.68, 12), (0.39, 0.52, 22)),
    tower=(0.45, 20, 54),
    mast=(0.66, 40),
)
KING_GEORGE_V = ShipProfile(
    "King George V class",
    turrets=(Turret(0.17, 1, width=32), Turret(0.27, 1, True, 24), Turret(0.80, -1, width=32)),
    funnels=(Funnel(0.50, 16, 22, rake=2), Funnel(0.62, 16, 22, rake=2)),
    decks=((0.34, 0.69, 12), (0.36, 0.48, 24)),
    tower=(0.40, 18, 38),
    mast=(0.45, 52),
)
RICHELIEU = ShipProfile(
    "Richelieu class",
    turrets=(Turret(0.15, 1, width=32), Turret(0.27, 1, True, 32)),
    funnels=(Funnel(0.62, 18, 30, rake=10),),
    decks=((0.40, 0.73, 12), (0.42, 0.55, 24)),
    tower=(0.47, 16, 48),
    mast=(0.64, 56),
)

NATIONS: tuple[Nation, ...] = (
    Nation("germany", "Germany", BISMARCK, _flag_germany),
    Nation("usa", "USA", IOWA, _flag_usa),
    Nation("japan", "Japan", YAMATO, _flag_japan),
    Nation("uk", "United Kingdom", KING_GEORGE_V, _flag_uk),
    Nation("france", "France", RICHELIEU, _flag_france),
)


def nation_by_key(key: str) -> Nation:
    for nation in NATIONS:
        if nation.key == key:
            return nation
    raise KeyError(key)


def flag_surface(nation: Nation, size: tuple[int, int]) -> pygame.Surface:
    surface = pygame.Surface(size)
    nation.draw_flag(surface)
    return surface

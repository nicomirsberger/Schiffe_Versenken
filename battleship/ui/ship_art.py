"""Procedurally drawn battleship illustrations with smoke, fire and sinking."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from battleship.ui import theme
from battleship.ui.nations import Nation, ShipProfile, flag_surface

SHIP_LENGTH = 340
HULL_HEIGHT = 16
MARGIN = 12  # free space around the ship on its surface
SURFACE_HEIGHT = 112
WATERLINE = 96  # y of the waterline on the ship surface


@dataclass
class ShipDrawing:
    """A rendered ship facing right plus anchor points for effects."""

    surface: pygame.Surface
    funnel_tops: list[tuple[int, int]]
    deck_points: list[tuple[int, int]]  # where fires can break out
    flagstaff_top: tuple[int, int]


def _deck_y(x: float) -> float:
    """Height of the main deck (sheer line rises towards the bow)."""
    rise = max(0.0, (x - 0.7) / 0.3) ** 2 * 7
    return WATERLINE - HULL_HEIGHT - rise


def render_ship(profile: ShipProfile) -> ShipDrawing:
    width = SHIP_LENGTH + 2 * MARGIN
    s = pygame.Surface((width, SURFACE_HEIGHT), pygame.SRCALPHA)

    def px(frac: float) -> int:
        return round(MARGIN + frac * SHIP_LENGTH)

    # Hull: flat stern, rising sheer, raked bow, a bit of hull below the waterline.
    top = [(px(f / 20), _deck_y(f / 20)) for f in range(21)]
    hull = [(px(0.0), WATERLINE + 6), (px(0.0) + 2, top[0][1])]
    hull += top[1:-1]
    hull += [(px(1.0) + 6, top[-1][1] - 2), (px(0.96), WATERLINE + 6)]
    pygame.draw.polygon(s, theme.HULL, hull)
    pygame.draw.line(s, theme.HULL_DARK, (px(0.0), WATERLINE - 4), (px(0.97), WATERLINE - 4), 2)

    # Superstructure blocks.
    for start, end, height in profile.decks:
        base = _deck_y(start)
        pygame.draw.rect(s, theme.SUPERSTRUCTURE, (px(start), base - height, px(end) - px(start), height))
        pygame.draw.line(s, theme.HULL_DARK, (px(start), base - height), (px(end), base - height), 1)

    # Bridge tower: stacked, narrowing levels.
    tx, tw, th = profile.tower
    base = _deck_y(tx)
    levels = 4
    for i in range(levels):
        w = round(tw * (1 - i * 0.18))
        h = th / levels
        rect = pygame.Rect(px(tx) - w // 2, round(base - (i + 1) * h), w, math.ceil(h))
        pygame.draw.rect(s, theme.SUPERSTRUCTURE, rect)
        pygame.draw.line(s, theme.HULL_DARK, rect.topleft, rect.topright, 1)
    tower_top = base - th

    # Mast with a yard.
    mx, mh = profile.mast
    mast_base = _deck_y(mx) - 20
    pygame.draw.line(s, theme.HULL_DARK, (px(mx), mast_base), (px(mx), mast_base - mh + 20), 2)
    pygame.draw.line(
        s, theme.HULL_DARK, (px(mx) - 8, mast_base - mh + 30), (px(mx) + 8, mast_base - mh + 30), 2
    )

    # Funnels, raked towards the stern.
    funnel_tops = []
    for funnel in profile.funnels:
        fx, base = px(funnel.x), _deck_y(funnel.x) - 20
        half = funnel.width // 2
        top_y = base - funnel.height
        shape = [
            (fx - half, base),
            (fx + half, base),
            (fx + half - funnel.rake, top_y),
            (fx - half - funnel.rake, top_y),
        ]
        pygame.draw.polygon(s, theme.SUPERSTRUCTURE, shape)
        pygame.draw.rect(s, theme.HULL_DARK, (fx - half - funnel.rake, top_y, funnel.width, 3))
        funnel_tops.append((fx - funnel.rake, top_y))

    # Turrets with gun barrels.
    for turret in profile.turrets:
        tx_px = px(turret.x)
        base = _deck_y(turret.x) - (9 if turret.raised else 0)
        if turret.raised:
            pygame.draw.rect(s, theme.HULL, (tx_px - 9, base, 18, 9))
        body = pygame.Rect(tx_px - turret.width // 2, base - 11, turret.width, 11)
        pygame.draw.rect(s, theme.SUPERSTRUCTURE, body, border_radius=3)
        muzzle_x = body.right + 22 if turret.facing > 0 else body.left - 22
        start_x = body.right - 2 if turret.facing > 0 else body.left + 2
        pygame.draw.line(s, theme.HULL_DARK, (start_x, body.centery - 1), (muzzle_x, body.centery - 3), 3)

    deck_points = [(px(f), round(_deck_y(f)) - 4) for f in (0.12, 0.3, 0.45, 0.58, 0.7, 0.88)]
    flagstaff_top = (px(0.015), round(_deck_y(0.0)) - 30)
    pygame.draw.line(s, theme.HULL_DARK, (px(0.015), _deck_y(0.0)), flagstaff_top, 2)
    _ = tower_top
    return ShipDrawing(s, funnel_tops, deck_points, flagstaff_top)


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    age: float
    life: float
    size: float
    shade: int


class ShipScene:
    """The battleship of one side, drawn above that side's board.

    ``damage`` (0..1) controls the number of fires and the amount of smoke;
    once ``defeated`` is set, the ship slowly lists and sinks.
    """

    def __init__(self, nation: Nation, rect: pygame.Rect, facing_right: bool, seed: int = 0) -> None:
        self.nation = nation
        self.rect = pygame.Rect(rect)
        self.facing_right = facing_right
        self.drawing = render_ship(nation.ship)
        self.flag = flag_surface(nation, (30, 19))
        self.particles: list[Particle] = []
        self.random = random.Random(seed)
        self.damage = 0.0
        self.defeated = False
        self.sink = 0.0  # 0 = afloat, 1 = fully sunk
        self.time = 0.0

    # ------------------------------------------------------------- geometry
    @property
    def origin(self) -> tuple[int, int]:
        """Top-left of the ship surface inside the scene."""
        x = self.rect.centerx - self.drawing.surface.get_width() // 2
        y = self.rect.bottom - 22 - WATERLINE
        return x, y

    def to_scene(self, point: tuple[int, int]) -> tuple[float, float]:
        """Map a point on the (right-facing) ship surface to screen coordinates."""
        width = self.drawing.surface.get_width()
        x, y = point
        if not self.facing_right:
            x = width - x
        ox, oy = self.origin
        # Apply sinking: list towards the stern and move down.
        angle = math.radians(self._list_angle())
        cx, cy = width / 2, WATERLINE
        dx, dy = x - cx, y - cy
        if not self.facing_right:
            angle = -angle
        rx = cx + dx * math.cos(angle) + dy * math.sin(angle)
        ry = cy - dx * math.sin(angle) + dy * math.cos(angle)
        return ox + rx, oy + ry + self._sink_offset()

    def _list_angle(self) -> float:
        return 11 * self.sink

    def _sink_offset(self) -> float:
        return 38 * self.sink

    # --------------------------------------------------------------- update
    def update(self, dt: float, damage: float, defeated: bool) -> None:
        self.time += dt
        self.damage = damage
        self.defeated = defeated
        if defeated:
            self.sink = min(1.0, self.sink + dt / 3.5)

        def spawn(point: tuple[float, float], dark: bool) -> None:
            x, y = point
            self.particles.append(
                Particle(
                    x + self.random.uniform(-3, 3),
                    y,
                    self.random.uniform(-16, -8),  # wind blows to the left
                    self.random.uniform(-22, -12),
                    0.0,
                    self.random.uniform(1.6, 2.6) if dark else self.random.uniform(1.0, 1.6),
                    self.random.uniform(3, 5),
                    self.random.randint(35, 60) if dark else self.random.randint(150, 185),
                )
            )

        if self.sink < 1.0:
            for top in self.drawing.funnel_tops:
                if self.random.random() < dt * 3:
                    spawn(self.to_scene(top), dark=damage > 0.5)
        for point in self.fire_points():
            if self.random.random() < dt * 7:
                spawn(point, dark=True)

        for p in self.particles:
            p.age += dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.size += dt * 6
        self.particles = [p for p in self.particles if p.age < p.life and p.y > self.rect.top - 10]

    def fire_points(self) -> list[tuple[float, float]]:
        n_fires = round(self.damage * len(self.drawing.deck_points))
        if self.sink >= 1.0:
            return []
        points = self.drawing.deck_points
        order = [2, 4, 0, 5, 1, 3]  # spread the fires over the ship
        return [self.to_scene(points[i]) for i in order[:n_fires]]

    # ----------------------------------------------------------------- draw
    def draw(self, surface: pygame.Surface, fonts: dict[str, pygame.font.Font]) -> None:
        clip = surface.get_clip()
        surface.set_clip(self.rect)
        pygame.draw.rect(surface, theme.SKY_TOP, self.rect)

        # Ship (mirrored if it faces left, rotated while sinking).
        ship = self.drawing.surface
        if not self.facing_right:
            ship = pygame.transform.flip(ship, True, False)
        ox, oy = self.origin
        if self.sink > 0:
            angle = self._list_angle() if self.facing_right else -self._list_angle()
            rotated = pygame.transform.rotate(ship, angle)
            center = (ox + ship.get_width() / 2, oy + WATERLINE + self._sink_offset())
            pivot_offset = pygame.math.Vector2(0, ship.get_height() / 2 - WATERLINE).rotate(-angle)
            rect = rotated.get_rect(center=(center[0] + pivot_offset.x, center[1] + pivot_offset.y))
            surface.blit(rotated, rect)
        else:
            surface.blit(ship, (ox, oy))

        if self.sink < 0.95:
            self._draw_flag(surface)
        self._draw_fires(surface)

        # Sea in front of the ship.
        sea_top = oy + WATERLINE
        sea = pygame.Surface((self.rect.width, self.rect.bottom - sea_top), pygame.SRCALPHA)
        sea.fill((*theme.SEA, 235))
        surface.blit(sea, (self.rect.left, sea_top))
        for i in range(0, self.rect.width, 22):
            phase = self.time * 2 + i * 0.3
            y = sea_top + 5 + 2 * math.sin(phase)
            x = self.rect.left + i + 6 * math.sin(phase * 0.5)
            pygame.draw.line(surface, theme.SEA_LINE, (x, y), (x + 10, y), 2)

        self._draw_smoke(surface)

        label = f"{self.nation.name} - {self.nation.ship.class_name}"
        if self.sink >= 1.0:
            label += "  (sunk)"
        text = fonts["small"].render(label, True, theme.TEXT_DIM)
        surface.blit(text, (self.rect.left + 8, self.rect.top + 6))
        surface.set_clip(clip)
        pygame.draw.rect(surface, theme.GRID_LINE, self.rect, width=1, border_radius=6)

    def _draw_flag(self, surface: pygame.Surface) -> None:
        sx, sy = self.to_scene(self.drawing.flagstaff_top)
        w, h = self.flag.get_size()
        direction = -1 if self.facing_right else 1  # flag streams away from the bow
        for col in range(w):
            wave = math.sin(self.time * 6 + col * 0.35) * 2 * (col / w)
            x = sx + direction * col - (1 if direction < 0 else 0)
            surface.blit(
                self.flag, (x, sy + wave), area=pygame.Rect(col if direction > 0 else w - 1 - col, 0, 1, h)
            )

    def _draw_fires(self, surface: pygame.Surface) -> None:
        for i, (x, y) in enumerate(self.fire_points()):
            flicker = math.sin(self.time * 18 + i * 1.7)
            radius = 5 + 1.5 * flicker
            pygame.draw.circle(surface, theme.FIRE, (x, y - 2), radius)
            pygame.draw.circle(surface, theme.FIRE_CORE, (x, y - 1), radius * 0.5)

    def _draw_smoke(self, surface: pygame.Surface) -> None:
        for p in self.particles:
            alpha = int(170 * (1 - p.age / p.life))
            size = int(p.size * 2)
            puff = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(puff, (p.shade, p.shade, p.shade, alpha), (size // 2, size // 2), size // 2)
            surface.blit(puff, (p.x - size / 2, p.y - size / 2))

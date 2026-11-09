"""Drawing of a single board."""

from __future__ import annotations

import numpy as np
import pygame

from battleship.game.board import CellState, Coord, Ship
from battleship.ui import theme


class BoardView:
    """Maps between screen pixels and board cells and draws a board."""

    def __init__(self, origin: tuple[int, int], size: int, title: str) -> None:
        self.origin = origin
        self.size = size
        self.title = title
        self.rect = pygame.Rect(origin[0], origin[1], size * theme.CELL, size * theme.CELL)

    def cell_rect(self, cell: Coord) -> pygame.Rect:
        row, col = cell
        return pygame.Rect(
            self.origin[0] + col * theme.CELL, self.origin[1] + row * theme.CELL, theme.CELL, theme.CELL
        )

    def cell_at(self, pos: tuple[int, int]) -> Coord | None:
        if not self.rect.collidepoint(pos):
            return None
        col = (pos[0] - self.origin[0]) // theme.CELL
        row = (pos[1] - self.origin[1]) // theme.CELL
        return int(row), int(col)

    # ------------------------------------------------------------------ drawing
    def draw_grid(self, surface: pygame.Surface, fonts: dict[str, pygame.font.Font]) -> None:
        title = fonts["heading"].render(self.title, True, theme.TEXT)
        surface.blit(title, (self.origin[0], self.origin[1] - 62))
        for i in range(self.size):
            letter = fonts["small"].render(chr(65 + i), True, theme.TEXT_DIM)
            cx = self.origin[0] + i * theme.CELL + theme.CELL // 2
            surface.blit(letter, letter.get_rect(center=(cx, self.origin[1] - 14)))
            number = fonts["small"].render(str(i + 1), True, theme.TEXT_DIM)
            cy = self.origin[1] + i * theme.CELL + theme.CELL // 2
            surface.blit(number, number.get_rect(center=(self.origin[0] - 18, cy)))
        for row in range(self.size):
            for col in range(self.size):
                color = theme.WATER if (row + col) % 2 == 0 else theme.WATER_DARK
                pygame.draw.rect(surface, color, self.cell_rect((row, col)))
        for i in range(self.size + 1):
            x = self.origin[0] + i * theme.CELL
            y = self.origin[1] + i * theme.CELL
            pygame.draw.line(surface, theme.GRID_LINE, (x, self.rect.top), (x, self.rect.bottom))
            pygame.draw.line(surface, theme.GRID_LINE, (self.rect.left, y), (self.rect.right, y))

    def draw_ships(self, surface: pygame.Surface, ships: list[Ship], alpha: int = 255) -> None:
        for ship in ships:
            rows = [r for r, _ in ship.cells]
            cols = [c for _, c in ship.cells]
            top_left = self.cell_rect((min(rows), min(cols)))
            bottom_right = self.cell_rect((max(rows), max(cols)))
            rect = top_left.union(bottom_right).inflate(-8, -8)
            layer = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(layer, (*theme.SHIP, alpha), layer.get_rect(), border_radius=12)
            pygame.draw.rect(layer, (*theme.SHIP_BORDER, alpha), layer.get_rect(), width=2, border_radius=12)
            surface.blit(layer, rect)

    def draw_shots(self, surface: pygame.Surface, grid: np.ndarray, last_shot: Coord | None = None) -> None:
        for row in range(self.size):
            for col in range(self.size):
                state = grid[row, col]
                rect = self.cell_rect((row, col))
                if state == CellState.MISS:
                    pygame.draw.circle(surface, theme.MISS, rect.center, 5)
                elif state in (CellState.HIT, CellState.SUNK):
                    color = theme.SUNK if state == CellState.SUNK else theme.HIT
                    inner = rect.inflate(-10, -10)
                    if state == CellState.SUNK:
                        pygame.draw.rect(surface, (*theme.SUNK,), rect.inflate(-4, -4), border_radius=6)
                    pygame.draw.line(
                        surface,
                        color if state == CellState.HIT else theme.TEXT,
                        inner.topleft,
                        inner.bottomright,
                        4,
                    )
                    pygame.draw.line(
                        surface,
                        color if state == CellState.HIT else theme.TEXT,
                        inner.topright,
                        inner.bottomleft,
                        4,
                    )
        if last_shot is not None:
            pygame.draw.rect(surface, theme.ACCENT, self.cell_rect(last_shot), width=3)

    def draw_heatmap(self, surface: pygame.Surface, scores: np.ndarray) -> None:
        """Overlay the AI's score map (values in [0, 1])."""
        peak = scores.max()
        if peak <= 0:
            return
        layer = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        for row in range(self.size):
            for col in range(self.size):
                value = float(scores[row, col] / peak)
                if value <= 0.02:
                    continue
                color = tuple(
                    int(lo + (hi - lo) * value)
                    for lo, hi in zip(theme.HEAT_LOW, theme.HEAT_HIGH, strict=True)
                )
                rect = self.cell_rect((row, col)).move(-self.origin[0], -self.origin[1])
                pygame.draw.rect(layer, (*color, int(60 + 150 * value)), rect)
        surface.blit(layer, self.origin)

    def draw_preview(self, surface: pygame.Surface, cells: tuple[Coord, ...], valid: bool) -> None:
        color = theme.VALID if valid else theme.INVALID
        for cell in cells:
            if 0 <= cell[0] < self.size and 0 <= cell[1] < self.size:
                layer = pygame.Surface((theme.CELL, theme.CELL), pygame.SRCALPHA)
                layer.fill((*color, 140))
                surface.blit(layer, self.cell_rect(cell))

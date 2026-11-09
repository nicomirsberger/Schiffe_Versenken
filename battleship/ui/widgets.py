"""Small reusable UI elements."""

from __future__ import annotations

import pygame

from battleship.ui import theme


class Button:
    def __init__(self, rect: tuple[int, int, int, int], label: str, sublabel: str = "") -> None:
        self.rect = pygame.Rect(rect)
        self.label = label
        self.sublabel = sublabel
        self.enabled = True

    def hovered(self, pos: tuple[int, int]) -> bool:
        return self.enabled and self.rect.collidepoint(pos)

    def draw(
        self, surface: pygame.Surface, fonts: dict[str, pygame.font.Font], mouse: tuple[int, int]
    ) -> None:
        color = theme.BUTTON_HOVER if self.hovered(mouse) else theme.BUTTON
        if not self.enabled:
            color = theme.PANEL
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, theme.GRID_LINE, self.rect, width=1, border_radius=8)
        text_color = theme.TEXT if self.enabled else theme.TEXT_DIM
        label = fonts["button"].render(self.label, True, text_color)
        if self.sublabel:
            sub = fonts["small"].render(self.sublabel, True, theme.TEXT_DIM)
            surface.blit(label, label.get_rect(midtop=(self.rect.centerx, self.rect.y + 10)))
            surface.blit(sub, sub.get_rect(midbottom=(self.rect.centerx, self.rect.bottom - 10)))
        else:
            surface.blit(label, label.get_rect(center=self.rect.center))

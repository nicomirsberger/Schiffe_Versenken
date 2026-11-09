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


class Dropdown:
    """A select box. Options that are ``disabled`` are shown but cannot be picked."""

    ROW = 38

    def __init__(
        self,
        rect: tuple[int, int, int, int],
        label: str,
        options: list[tuple[str, str]],
        selected: str,
        icons: dict[str, pygame.Surface] | None = None,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.label = label
        self.options = options
        self.selected = selected
        self.icons = icons or {}
        self.disabled: set[str] = set()
        self.open = False

    def option_rect(self, index: int) -> pygame.Rect:
        return pygame.Rect(self.rect.x, self.rect.bottom + index * self.ROW, self.rect.width, self.ROW)

    def handle_click(self, pos: tuple[int, int]) -> bool:
        """Process a click. Returns ``True`` if the dropdown consumed it."""
        if self.open:
            for index, (key, _) in enumerate(self.options):
                if self.option_rect(index).collidepoint(pos):
                    if key not in self.disabled:
                        self.selected = key
                        self.open = False
                    return True
            self.open = False
            return self.rect.collidepoint(pos)
        if self.rect.collidepoint(pos):
            self.open = True
            return True
        return False

    def _draw_row(self, surface, fonts, key: str, text: str, rect: pygame.Rect, color) -> None:
        x = rect.x + 12
        icon = self.icons.get(key)
        if icon is not None:
            surface.blit(icon, icon.get_rect(midleft=(x, rect.centery)))
            x += icon.get_width() + 10
        rendered = fonts["text"].render(text, True, color)
        surface.blit(rendered, rendered.get_rect(midleft=(x, rect.centery)))

    def draw(
        self, surface: pygame.Surface, fonts: dict[str, pygame.font.Font], mouse: tuple[int, int]
    ) -> None:
        caption = fonts["small"].render(self.label, True, theme.TEXT_DIM)
        surface.blit(caption, (self.rect.x, self.rect.y - 22))
        color = theme.BUTTON_HOVER if self.rect.collidepoint(mouse) or self.open else theme.BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, theme.GRID_LINE, self.rect, width=1, border_radius=8)
        text = dict(self.options)[self.selected]
        self._draw_row(surface, fonts, self.selected, text, self.rect, theme.TEXT)
        arrow_x, arrow_y = self.rect.right - 22, self.rect.centery
        points = [(arrow_x - 6, arrow_y - 3), (arrow_x + 6, arrow_y - 3), (arrow_x, arrow_y + 4)]
        if self.open:
            points = [(arrow_x - 6, arrow_y + 3), (arrow_x + 6, arrow_y + 3), (arrow_x, arrow_y - 4)]
        pygame.draw.polygon(surface, theme.TEXT, points)

    def draw_options(
        self, surface: pygame.Surface, fonts: dict[str, pygame.font.Font], mouse: tuple[int, int]
    ) -> None:
        """Draw the open list (call last so it is on top of everything else)."""
        if not self.open:
            return
        for index, (key, text) in enumerate(self.options):
            rect = self.option_rect(index)
            disabled = key in self.disabled
            background = theme.PANEL
            if rect.collidepoint(mouse) and not disabled:
                background = theme.BUTTON_HOVER
            elif key == self.selected:
                background = theme.BUTTON
            pygame.draw.rect(surface, background, rect)
            pygame.draw.rect(surface, theme.GRID_LINE, rect, width=1)
            label = f"{text}  (already chosen)" if disabled else text
            self._draw_row(surface, fonts, key, label, rect, theme.TEXT_DIM if disabled else theme.TEXT)

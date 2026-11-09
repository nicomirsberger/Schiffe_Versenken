"""Main window: menu -> fleet setup -> battle -> result."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
import pygame

from battleship.ai.factory import DIFFICULTIES, create_agent
from battleship.config import DEFAULT_CONFIG, SHIP_NAMES, GameConfig
from battleship.game.board import Board, Coord, Ship, ShotOutcome, ShotResult
from battleship.game.match import Game, Player
from battleship.game.placement import can_place, random_fleet, ship_cells
from battleship.ui import theme
from battleship.ui.board_view import BoardView
from battleship.ui.widgets import Button


class Screen(Enum):
    MENU = auto()
    SETUP = auto()
    BATTLE = auto()
    RESULT = auto()


class App:
    def __init__(self, config: GameConfig = DEFAULT_CONFIG, seed: int | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Battleship AI")
        self.surface = pygame.display.set_mode(theme.WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.fonts = {
            "title": pygame.font.SysFont("arial", 46, bold=True),
            "heading": pygame.font.SysFont("arial", 24, bold=True),
            "button": pygame.font.SysFont("arial", 22, bold=True),
            "text": pygame.font.SysFont("arial", 20),
            "small": pygame.font.SysFont("arial", 16),
        }
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.left = BoardView(theme.LEFT_BOARD, config.size, "Your fleet")
        self.right = BoardView(theme.RIGHT_BOARD, config.size, "Enemy waters")
        self.menu_buttons = [
            Button((330, 250 + i * 95, 520, 80), d.label, d.description) for i, d in enumerate(DIFFICULTIES)
        ]
        self.setup_buttons = {
            "random": Button((680, 560, 150, 46), "Random"),
            "undo": Button((845, 560, 150, 46), "Undo"),
            "start": Button((680, 620, 315, 52), "Start battle"),
        }
        self.result_button = Button((490, 470, 200, 56), "Play again")
        self.running = True
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.screen = Screen.MENU
        self.difficulty = DIFFICULTIES[-1]
        self.placed: list[Ship] = []
        self.horizontal = True
        self.game: Game | None = None
        self.message = ""
        self.show_heatmap = False
        self.last_human_shot: Coord | None = None
        self.last_ai_shot: Coord | None = None
        self.ai_ready_at = 0
        self._heat_key: tuple[int, int] | None = None
        self._heat = np.zeros((self.config.size, self.config.size))

    @property
    def ships_to_place(self) -> list[int]:
        return list(self.config.fleet[len(self.placed) :])

    def start_setup(self, index: int) -> None:
        self.difficulty = DIFFICULTIES[index]
        self.placed = []
        self.screen = Screen.SETUP
        self.message = "Place your ships."

    def start_battle(self) -> None:
        agent, warning = create_agent(self.difficulty.key)
        self.game = Game(
            Board(self.placed, self.config),
            Board(random_fleet(self.config, self.rng), self.config),
            agent,
            self.rng,
            starting_player=Player.HUMAN,
        )
        self.screen = Screen.BATTLE
        self.message = warning or "Your turn - click on the enemy waters to fire."

    # ------------------------------------------------------------------ events
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self.handle_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3 and self.screen is Screen.SETUP:
                self.horizontal = not self.horizontal
            elif event.button == 1:
                self.handle_click(event.pos)

    def handle_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self.reset()
        elif key == pygame.K_h and self.screen in (Screen.BATTLE, Screen.RESULT):
            self.show_heatmap = not self.show_heatmap
        elif self.screen is Screen.MENU and pygame.K_1 <= key <= pygame.K_4:
            self.start_setup(key - pygame.K_1)
        elif self.screen is Screen.SETUP:
            if key == pygame.K_r:
                self.horizontal = not self.horizontal
            elif key == pygame.K_SPACE:
                self.placed = random_fleet(self.config, self.rng)
            elif key == pygame.K_BACKSPACE and self.placed:
                self.placed.pop()
            elif key == pygame.K_RETURN and not self.ships_to_place:
                self.start_battle()
        elif self.screen is Screen.RESULT and key == pygame.K_RETURN:
            self.reset()

    def handle_click(self, pos: tuple[int, int]) -> None:
        if self.screen is Screen.MENU:
            for index, button in enumerate(self.menu_buttons):
                if button.hovered(pos):
                    self.start_setup(index)
        elif self.screen is Screen.SETUP:
            self.handle_setup_click(pos)
        elif self.screen is Screen.BATTLE:
            self.handle_battle_click(pos)
        elif self.screen is Screen.RESULT and self.result_button.hovered(pos):
            self.reset()

    def handle_setup_click(self, pos: tuple[int, int]) -> None:
        buttons = self.setup_buttons
        if buttons["random"].hovered(pos):
            self.placed = random_fleet(self.config, self.rng)
        elif buttons["undo"].hovered(pos) and self.placed:
            self.placed.pop()
        elif buttons["start"].hovered(pos) and not self.ships_to_place:
            self.start_battle()
        else:
            ship = self.preview_ship(pos)
            if ship is not None and can_place(self.placed, ship, self.config):
                self.placed.append(ship)

    def handle_battle_click(self, pos: tuple[int, int]) -> None:
        assert self.game is not None
        cell = self.right.cell_at(pos)
        if cell is None or self.game.current is not Player.HUMAN or self.game.ai_board.is_shot(cell):
            return
        outcome = self.game.human_fire(cell)
        self.last_human_shot = cell
        self.message = self.describe(outcome, "You")
        self.ai_ready_at = pygame.time.get_ticks() + theme.AI_DELAY_MS
        self.check_game_over()

    def preview_ship(self, pos: tuple[int, int]) -> Ship | None:
        cell = self.left.cell_at(pos)
        if cell is None or not self.ships_to_place:
            return None
        return Ship(ship_cells(cell, self.ships_to_place[0], self.horizontal))

    # ------------------------------------------------------------------ update
    def update(self) -> None:
        game = self.game
        if self.screen is not Screen.BATTLE or game is None or game.is_over:
            return
        if game.current is Player.AI and pygame.time.get_ticks() >= self.ai_ready_at:
            outcome = game.ai_fire()
            self.last_ai_shot = outcome.coord
            self.message = self.describe(outcome, "The AI")
            self.ai_ready_at = pygame.time.get_ticks() + theme.AI_DELAY_MS
            self.check_game_over()

    def current_ai_scores(self) -> np.ndarray:
        """The AI's score map for the *current* state of the player's board.

        Agents store their scores when they choose a shot, so we let the agent
        evaluate the current board once (without firing) and cache the result.
        """
        assert self.game is not None
        board = self.game.human_board
        key = (board.shots_fired, id(self.game))
        if self._heat_key != key:
            obs = board.observation()
            if obs.unknown_mask.any():
                self.game.agent.choose_shot(obs, np.random.default_rng(0))
            scores = self.game.agent.last_scores
            self._heat = np.zeros(obs.grid.shape) if scores is None else scores.copy()
            self._heat_key = key
        return self._heat

    def check_game_over(self) -> None:
        if self.game is not None and self.game.is_over:
            self.screen = Screen.RESULT

    @staticmethod
    def describe(outcome: ShotOutcome, shooter: str) -> str:
        cell = f"{chr(65 + outcome.coord[1])}{outcome.coord[0] + 1}"
        if outcome.result is ShotResult.MISS:
            return f"{shooter} fired at {cell}: miss."
        if outcome.result is ShotResult.HIT:
            return f"{shooter} fired at {cell}: HIT! Shoot again."
        name = SHIP_NAMES.get(outcome.ship.length if outcome.ship else 0, "ship")
        return f"{shooter} fired at {cell}: {name} SUNK!"

    # ------------------------------------------------------------------ drawing
    def draw(self) -> None:
        self.surface.fill(theme.BACKGROUND)
        mouse = pygame.mouse.get_pos()
        if self.screen is Screen.MENU:
            self.draw_menu(mouse)
        elif self.screen is Screen.SETUP:
            self.draw_setup(mouse)
        else:
            self.draw_battle()
            if self.screen is Screen.RESULT:
                self.draw_result(mouse)
        pygame.display.flip()

    def text(
        self, content: str, pos: tuple[int, int], font: str = "text", color=theme.TEXT, center=False
    ) -> None:
        rendered = self.fonts[font].render(content, True, color)
        rect = rendered.get_rect(center=pos) if center else rendered.get_rect(topleft=pos)
        self.surface.blit(rendered, rect)

    def draw_menu(self, mouse: tuple[int, int]) -> None:
        cx = theme.WINDOW_SIZE[0] // 2
        self.text("BATTLESHIP AI", (cx, 110), "title", theme.ACCENT, center=True)
        self.text("Sink the enemy fleet before the AI sinks yours.", (cx, 170), center=True)
        self.text(
            "Choose the strength of your opponent (or press 1-4):", (cx, 215), "small", theme.TEXT_DIM, True
        )
        for button in self.menu_buttons:
            button.draw(self.surface, self.fonts, mouse)
        fleet = ", ".join(
            f"{self.config.fleet_count(n)}x {SHIP_NAMES.get(n, n)} ({n})"
            for n in sorted(set(self.config.fleet), reverse=True)
        )
        self.text(f"Fleet: {fleet}", (cx, 660), "small", theme.TEXT_DIM, center=True)
        rule = "Ships may not touch." if not self.config.allow_touching else "Ships may touch."
        self.text(f"{rule}  A hit earns another shot.", (cx, 685), "small", theme.TEXT_DIM, center=True)

    def draw_header(self) -> None:
        self.text("BATTLESHIP AI", (80, 22), "heading", theme.ACCENT)
        self.text(f"Opponent: {self.difficulty.label}", (300, 26), "text", theme.TEXT_DIM)
        self.text(self.message, (80, 62), "text")

    def draw_setup(self, mouse: tuple[int, int]) -> None:
        self.draw_header()
        self.left.draw_grid(self.surface, self.fonts)
        self.left.draw_ships(self.surface, self.placed)
        ship = self.preview_ship(mouse)
        if ship is not None:
            self.left.draw_preview(self.surface, ship.cells, can_place(self.placed, ship, self.config))

        x, y = theme.RIGHT_BOARD
        self.text("Ships to place", (x, y - 62), "heading")
        remaining = self.ships_to_place
        for i, length in enumerate(remaining[:6]):
            label = f"{SHIP_NAMES.get(length, 'Ship')} ({length})" + ("  <- next" if i == 0 else "")
            self.text(label, (x, y + i * 30), "text", theme.ACCENT if i == 0 else theme.TEXT)
            for j in range(length):
                pygame.draw.rect(
                    self.surface, theme.SHIP, (x + 260 + j * 22, y + 4 + i * 30, 18, 18), border_radius=4
                )
        if len(remaining) > 6:
            self.text(f"... and {len(remaining) - 6} more", (x, y + 6 * 30), "small", theme.TEXT_DIM)
        if not remaining:
            self.text("All ships placed - start the battle!", (x, y), "text", theme.VALID)
        help_lines = [
            "Left click: place ship   |   R / right click: rotate",
            "Space: random fleet   |   Backspace: undo   |   Enter: start",
        ]
        for i, line in enumerate(help_lines):
            self.text(line, (x, 480 + i * 24), "small", theme.TEXT_DIM)
        self.setup_buttons["start"].enabled = not remaining
        self.setup_buttons["undo"].enabled = bool(self.placed)
        for button in self.setup_buttons.values():
            button.draw(self.surface, self.fonts, mouse)

    def draw_battle(self) -> None:
        assert self.game is not None
        game = self.game
        self.draw_header()
        turn = "Your turn" if game.current is Player.HUMAN else "AI is aiming ..."
        if not game.is_over:
            self.text(turn, (900, 26), "heading", theme.VALID if game.current is Player.HUMAN else theme.HIT)

        self.left.draw_grid(self.surface, self.fonts)
        self.left.draw_ships(self.surface, game.human_board.ships)
        if self.show_heatmap:
            self.left.draw_heatmap(self.surface, self.current_ai_scores())
        self.left.draw_shots(self.surface, game.human_board.observation().grid, self.last_ai_shot)

        self.right.draw_grid(self.surface, self.fonts)
        if game.is_over:
            self.right.draw_ships(self.surface, game.ai_board.ships, alpha=150)
        self.right.draw_shots(self.surface, game.ai_board.observation().grid, self.last_human_shot)

        self.draw_fleet_status(game.human_board, self.left.origin)
        self.draw_fleet_status(game.ai_board, self.right.origin)
        heat = "on" if self.show_heatmap else "off"
        hint = f"H: show what the AI thinks ({heat})"
        if self.show_heatmap:
            hint += " - red = AI expects a ship there"
        self.text(f"{hint}   |   Esc: menu", (80, 690), "small", theme.TEXT_DIM)

    def draw_fleet_status(self, board: Board, origin: tuple[int, int]) -> None:
        x, y = origin[0], origin[1] + self.config.size * theme.CELL + 18
        afloat = list(board.remaining_ship_lengths())
        for length in self.config.fleet:
            sunk = length not in afloat
            if not sunk:
                afloat.remove(length)
            color = theme.SUNK if sunk else theme.SHIP
            for j in range(length):
                pygame.draw.rect(self.surface, color, (x + j * 12, y, 10, 10), border_radius=2)
            x += length * 12 + 12
        self.text(f"Shots taken: {board.shots_fired}", (origin[0], y + 22), "small", theme.TEXT_DIM)

    def draw_result(self, mouse: tuple[int, int]) -> None:
        assert self.game is not None
        overlay = pygame.Surface(theme.WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.surface.blit(overlay, (0, 0))
        won = self.game.winner is Player.HUMAN
        cx = theme.WINDOW_SIZE[0] // 2
        title = "VICTORY!" if won else "DEFEAT"
        self.text(title, (cx, 300), "title", theme.VALID if won else theme.SUNK, center=True)
        shots = self.game.ai_board.shots_fired if won else self.game.human_board.shots_fired
        who = "You" if won else f"The {self.difficulty.label} AI"
        self.text(f"{who} sank the whole fleet with {shots} shots.", (cx, 370), center=True)
        self.text("Press H to see the AI's last probability map.", (cx, 410), "small", theme.TEXT_DIM, True)
        self.result_button.draw(self.surface, self.fonts, mouse)

    # ------------------------------------------------------------------ loop
    def run(self, max_frames: int | None = None) -> None:
        frames = 0
        while self.running and (max_frames is None or frames < max_frames):
            for event in pygame.event.get():
                self.handle_event(event)
            self.update()
            self.draw()
            self.clock.tick(60)
            frames += 1
        pygame.quit()


def run() -> None:
    App().run()

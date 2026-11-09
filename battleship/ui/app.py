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
from battleship.ui.nations import NATIONS, Nation, flag_surface, nation_by_key
from battleship.ui.ship_art import ShipScene
from battleship.ui.widgets import Button, Dropdown

RESULT_COLORS = {ShotResult.MISS: theme.MISS, ShotResult.HIT: theme.HIT, ShotResult.SUNK: theme.SUNK}


class Screen(Enum):
    MENU = auto()
    SETUP = auto()
    BATTLE = auto()
    RESULT = auto()


def cell_name(cell: Coord) -> str:
    return f"{chr(65 + cell[1])}{cell[0] + 1}"


def describe(outcome: ShotOutcome) -> str:
    """Short text for the shot log, e.g. ``"J3  Cruiser SUNK"``."""
    if outcome.result is ShotResult.MISS:
        return f"{cell_name(outcome.coord)}  miss"
    if outcome.result is ShotResult.HIT:
        return f"{cell_name(outcome.coord)}  HIT"
    name = SHIP_NAMES.get(outcome.ship.length if outcome.ship else 0, "Ship")
    return f"{cell_name(outcome.coord)}  {name} SUNK"


class App:
    def __init__(self, config: GameConfig = DEFAULT_CONFIG, seed: int | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Battleship AI")
        self.surface = pygame.display.set_mode(theme.WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.fonts = {
            "title": pygame.font.SysFont("arial", 46, bold=True),
            "heading": pygame.font.SysFont("arial", 22, bold=True),
            "button": pygame.font.SysFont("arial", 21, bold=True),
            "text": pygame.font.SysFont("arial", 19),
            "log": pygame.font.SysFont("arial", 18, bold=True),
            "small": pygame.font.SysFont("arial", 15),
        }
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.left = BoardView(theme.LEFT_BOARD, config.size, "Your fleet")
        self.right = BoardView(theme.RIGHT_BOARD, config.size, "Enemy waters")

        options = [(n.key, n.name) for n in NATIONS]
        icons = {n.key: flag_surface(n, (30, 19)) for n in NATIONS}
        self.player_select = Dropdown((250, 205, 300, 46), "YOUR NATION", options, "germany", icons)
        self.enemy_select = Dropdown((630, 205, 300, 46), "OPPONENT NATION", options, "usa", icons)
        self.menu_buttons = [
            Button((330, 455 + i * 84, 520, 72), d.label, d.description) for i, d in enumerate(DIFFICULTIES)
        ]
        self.setup_buttons = {
            "random": Button((730, 640, 170, 46), "Random"),
            "undo": Button((915, 640, 170, 46), "Undo"),
            "start": Button((730, 700, 355, 52), "Start battle"),
        }
        self.result_button = Button((490, 500, 200, 56), "Play again")
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
        self.ai_ready_at = 0
        self.flashes: list[tuple[BoardView, Coord, int, tuple[int, int, int]]] = []
        self._heat_key: tuple[int, int] | None = None
        self._heat = np.zeros((self.config.size, self.config.size))
        self.update_nation_options()
        self.build_scenes()

    @property
    def player_nation(self) -> Nation:
        return nation_by_key(self.player_select.selected)

    @property
    def enemy_nation(self) -> Nation:
        return nation_by_key(self.enemy_select.selected)

    def update_nation_options(self) -> None:
        """A nation cannot fight against itself."""
        self.player_select.disabled = {self.enemy_select.selected}
        self.enemy_select.disabled = {self.player_select.selected}

    def build_scenes(self) -> None:
        def scene_rect(board: BoardView) -> pygame.Rect:
            return pygame.Rect(
                board.origin[0] - 20, theme.SCENE_TOP, board.rect.width + 40, theme.SCENE_HEIGHT
            )

        self.player_scene = ShipScene(self.player_nation, scene_rect(self.left), facing_right=True, seed=1)
        self.enemy_scene = ShipScene(self.enemy_nation, scene_rect(self.right), facing_right=False, seed=2)

    @property
    def ships_to_place(self) -> list[int]:
        return list(self.config.fleet[len(self.placed) :])

    def start_setup(self, index: int) -> None:
        self.difficulty = DIFFICULTIES[index]
        self.player_select.open = self.enemy_select.open = False
        self.placed = []
        self.build_scenes()
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
        self.message = warning or ""

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
            self.handle_menu_click(pos)
        elif self.screen is Screen.SETUP:
            self.handle_setup_click(pos)
        elif self.screen is Screen.BATTLE:
            self.handle_battle_click(pos)
        elif self.screen is Screen.RESULT and self.result_button.hovered(pos):
            self.reset()

    def handle_menu_click(self, pos: tuple[int, int]) -> None:
        # An open list lies on top of everything else, so it gets the click first.
        for dropdown in (self.player_select, self.enemy_select):
            if dropdown.open:
                dropdown.handle_click(pos)
                self.update_nation_options()
                self.build_scenes()
                return
        for dropdown, other in (
            (self.player_select, self.enemy_select),
            (self.enemy_select, self.player_select),
        ):
            if dropdown.handle_click(pos):
                other.open = False
                return
        for index, button in enumerate(self.menu_buttons):
            if button.hovered(pos):
                self.start_setup(index)

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
        self.add_flash(self.right, outcome)
        self.ai_ready_at = pygame.time.get_ticks() + theme.AI_DELAY_MS
        self.check_game_over()

    def preview_ship(self, pos: tuple[int, int]) -> Ship | None:
        cell = self.left.cell_at(pos)
        if cell is None or not self.ships_to_place:
            return None
        return Ship(ship_cells(cell, self.ships_to_place[0], self.horizontal))

    def add_flash(self, view: BoardView, outcome: ShotOutcome) -> None:
        self.flashes.append((view, outcome.coord, pygame.time.get_ticks(), RESULT_COLORS[outcome.result]))

    # ------------------------------------------------------------------ update
    def update(self, dt: float) -> None:
        game = self.game
        if self.screen in (Screen.BATTLE, Screen.RESULT) and game is not None:
            self.player_scene.update(dt, self.damage(game.human_board), game.winner is Player.AI)
            self.enemy_scene.update(dt, self.damage(game.ai_board), game.winner is Player.HUMAN)
        else:
            self.player_scene.update(dt, 0.0, False)
            self.enemy_scene.update(dt, 0.0, False)

        now = pygame.time.get_ticks()
        self.flashes = [f for f in self.flashes if now - f[2] < theme.FLASH_MS]

        if self.screen is not Screen.BATTLE or game is None or game.is_over:
            return
        if game.current is Player.AI and now >= self.ai_ready_at:
            outcome = game.ai_fire()
            self.add_flash(self.left, outcome)
            self.ai_ready_at = now + theme.AI_REPEAT_DELAY_MS
            self.check_game_over()

    def damage(self, board: Board) -> float:
        """Share of the fleet that has been sunk (drives smoke and fire)."""
        return 1 - len(board.remaining_ship_lengths()) / len(self.config.fleet)

    def check_game_over(self) -> None:
        if self.game is not None and self.game.is_over:
            self.screen = Screen.RESULT

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
        self, content: str, pos, font: str = "text", color=theme.TEXT, center=False, right=False
    ) -> None:
        rendered = self.fonts[font].render(content, True, color)
        if center:
            rect = rendered.get_rect(center=pos)
        elif right:
            rect = rendered.get_rect(topright=pos)
        else:
            rect = rendered.get_rect(topleft=pos)
        self.surface.blit(rendered, rect)

    def draw_menu(self, mouse: tuple[int, int]) -> None:
        cx = theme.WINDOW_SIZE[0] // 2
        self.text("BATTLESHIP AI", (cx, 70), "title", theme.ACCENT, center=True)
        self.text("Sink the enemy fleet before the AI sinks yours.", (cx, 122), center=True)

        self.player_select.draw(self.surface, self.fonts, mouse)
        self.enemy_select.draw(self.surface, self.fonts, mouse)
        self.text("vs.", (cx, 228), "heading", theme.TEXT_DIM, center=True)

        for scene, x in ((self.player_scene, 160), (self.enemy_scene, 600)):
            scene.rect = pygame.Rect(x, 270, 420, theme.SCENE_HEIGHT)
            scene.draw(self.surface, self.fonts)

        self.text(
            "Choose the strength of your opponent (or press 1-4):", (cx, 425), "small", theme.TEXT_DIM, True
        )
        for button in self.menu_buttons:
            button.draw(self.surface, self.fonts, mouse)
        fleet = ", ".join(
            f"{self.config.fleet_count(n)}x {SHIP_NAMES.get(n, n)} ({n})"
            for n in sorted(set(self.config.fleet), reverse=True)
        )
        rule = "Ships may not touch." if not self.config.allow_touching else "Ships may touch."
        self.text(
            f"Fleet: {fleet}.  {rule}  A hit earns another shot.", (cx, 812), "small", theme.TEXT_DIM, True
        )

        # Open lists last, so they lie on top of everything else.
        self.player_select.draw_options(self.surface, self.fonts, mouse)
        self.enemy_select.draw_options(self.surface, self.fonts, mouse)

    def draw_header(self, status: str = "", status_color=theme.TEXT) -> None:
        self.text("BATTLESHIP AI", (90, 20), "heading", theme.ACCENT)
        info = f"{self.player_nation.name} vs. {self.enemy_nation.name}  |  Opponent: {self.difficulty.label}"
        self.text(info, (90, 52), "small", theme.TEXT_DIM)
        if status:
            self.text(status, (1090, 18), "heading", status_color, right=True)
        if self.message:
            self.text(self.message, (1090, 52), "small", theme.TEXT_DIM, right=True)

    def draw_scenes(self) -> None:
        for scene, board in ((self.player_scene, self.left), (self.enemy_scene, self.right)):
            scene.rect = pygame.Rect(
                board.origin[0] - 20, theme.SCENE_TOP, board.rect.width + 40, theme.SCENE_HEIGHT
            )
            scene.draw(self.surface, self.fonts)

    def draw_setup(self, mouse: tuple[int, int]) -> None:
        self.draw_header("Place your ships", theme.ACCENT)
        self.draw_scenes()
        self.left.draw_grid(self.surface, self.fonts)
        self.left.draw_ships(self.surface, self.placed)
        ship = self.preview_ship(mouse)
        if ship is not None:
            self.left.draw_preview(self.surface, ship.cells, can_place(self.placed, ship, self.config))

        x, y = theme.RIGHT_BOARD
        self.text("Ships to place", (x, y - 56), "heading")
        remaining = self.ships_to_place
        for i, length in enumerate(remaining[:7]):
            label = f"{SHIP_NAMES.get(length, 'Ship')} ({length})" + ("  <- next" if i == 0 else "")
            self.text(label, (x, y + i * 30), "text", theme.ACCENT if i == 0 else theme.TEXT)
            for j in range(length):
                pygame.draw.rect(
                    self.surface, theme.SHIP, (x + 240 + j * 22, y + 4 + i * 30, 18, 18), border_radius=4
                )
        if len(remaining) > 7:
            self.text(f"... and {len(remaining) - 7} more", (x, y + 7 * 30), "small", theme.TEXT_DIM)
        if not remaining:
            self.text("All ships placed - start the battle!", (x, y), "text", theme.VALID)
        help_lines = [
            "Left click: place ship   |   R / right click: rotate",
            "Space: random fleet   |   Backspace: undo   |   Enter: start",
        ]
        for i, line in enumerate(help_lines):
            self.text(line, (x, 572 + i * 24), "small", theme.TEXT_DIM)
        self.setup_buttons["start"].enabled = not remaining
        self.setup_buttons["undo"].enabled = bool(self.placed)
        for button in self.setup_buttons.values():
            button.draw(self.surface, self.fonts, mouse)

    def draw_battle(self) -> None:
        assert self.game is not None
        game = self.game
        if game.is_over:
            self.draw_header()
        elif game.current is Player.HUMAN:
            self.draw_header("Your turn - fire at the enemy waters", theme.VALID)
        else:
            self.draw_header(f"{self.enemy_nation.name} is aiming ...", theme.HIT)
        self.draw_scenes()

        last_ai = self.last_shot(Player.AI)
        last_human = self.last_shot(Player.HUMAN)

        self.left.draw_grid(self.surface, self.fonts)
        self.left.draw_ships(self.surface, game.human_board.ships)
        if self.show_heatmap:
            self.left.draw_heatmap(self.surface, self.current_ai_scores())
        self.left.draw_shots(self.surface, game.human_board.observation().grid, last_ai)

        self.right.draw_grid(self.surface, self.fonts)
        if game.is_over:
            self.right.draw_ships(self.surface, game.ai_board.ships, alpha=150)
        self.right.draw_shots(self.surface, game.ai_board.observation().grid, last_human)

        now = pygame.time.get_ticks()
        for view, cell, start, color in self.flashes:
            view.draw_flash(self.surface, cell, (now - start) / theme.FLASH_MS, color)

        self.draw_fleet_status(game.human_board, self.left)
        self.draw_fleet_status(game.ai_board, self.right)
        self.draw_log(
            Player.AI, f"{self.enemy_nation.name} fires at you", self.left, game.human_board.shots_fired
        )
        self.draw_log(Player.HUMAN, "Your shots", self.right, game.ai_board.shots_fired)

        heat = "on" if self.show_heatmap else "off"
        self.text(f"H: AI heatmap ({heat})   |   Esc: menu", (90, 818), "small", theme.TEXT_DIM)

    def last_shot(self, player: Player) -> Coord | None:
        assert self.game is not None
        for shooter, outcome in reversed(self.game.history):
            if shooter is player:
                return outcome.coord
        return None

    def recent_shots(self, player: Player, count: int = theme.LOG_ROWS) -> list[ShotOutcome]:
        assert self.game is not None
        shots = [outcome for shooter, outcome in self.game.history if shooter is player]
        return shots[::-1][:count]

    def draw_fleet_status(self, board: Board, view: BoardView) -> None:
        x, y = view.origin[0], view.rect.bottom + 12
        afloat = list(board.remaining_ship_lengths())
        for length in self.config.fleet:
            sunk = length not in afloat
            if not sunk:
                afloat.remove(length)
            color = theme.SUNK if sunk else theme.SHIP
            for j in range(length):
                pygame.draw.rect(self.surface, color, (x + j * 9, y, 7, 7), border_radius=2)
            x += length * 9 + 8

    def draw_log(self, player: Player, title: str, view: BoardView, total: int) -> None:
        box = pygame.Rect(view.origin[0], theme.LOG_TOP, view.rect.width, 118)
        pygame.draw.rect(self.surface, theme.PANEL, box, border_radius=8)
        pygame.draw.rect(self.surface, theme.GRID_LINE, box, width=1, border_radius=8)
        self.text(title, (box.x + 12, box.y + 7), "small", theme.TEXT_DIM)
        self.text(f"{total} shots", (box.right - 12, box.y + 7), "small", theme.TEXT_DIM, right=True)

        shots = self.recent_shots(player)
        if not shots:
            self.text("No shots yet.", (box.x + 12, box.y + 40), "text", theme.TEXT_DIM)
        for i, outcome in enumerate(shots):
            row = pygame.Rect(box.x + 6, box.y + 30 + i * 28, box.width - 12, 26)
            if i == 0:
                pygame.draw.rect(self.surface, theme.BUTTON, row, border_radius=6)
            color = RESULT_COLORS[outcome.result]
            icon = (row.x + 16, row.centery)
            if outcome.result is ShotResult.MISS:
                pygame.draw.circle(self.surface, color, icon, 4)
            else:
                pygame.draw.line(
                    self.surface, color, (icon[0] - 6, icon[1] - 6), (icon[0] + 6, icon[1] + 6), 3
                )
                pygame.draw.line(
                    self.surface, color, (icon[0] - 6, icon[1] + 6), (icon[0] + 6, icon[1] - 6), 3
                )
            text_color = {
                ShotResult.MISS: theme.TEXT,
                ShotResult.HIT: theme.HIT,
                ShotResult.SUNK: theme.SUNK_TEXT,
            }[outcome.result]
            if i > 0:
                text_color = tuple(int(c * 0.85) for c in text_color)
            self.text(describe(outcome), (row.x + 34, row.y + 3), "log", text_color)

    def draw_result(self, mouse: tuple[int, int]) -> None:
        assert self.game is not None
        overlay = pygame.Surface(theme.WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.surface.blit(overlay, (0, 0))
        won = self.game.winner is Player.HUMAN
        cx = theme.WINDOW_SIZE[0] // 2
        panel = pygame.Rect(cx - 300, 330, 600, 250)
        pygame.draw.rect(self.surface, theme.PANEL, panel, border_radius=12)
        pygame.draw.rect(self.surface, theme.GRID_LINE, panel, width=1, border_radius=12)
        title = "VICTORY!" if won else "DEFEAT"
        self.text(title, (cx, 375), "title", theme.VALID if won else theme.SUNK, center=True)
        winner, loser = (
            (self.player_nation, self.enemy_nation) if won else (self.enemy_nation, self.player_nation)
        )
        shots = self.game.ai_board.shots_fired if won else self.game.human_board.shots_fired
        self.text(f"{winner.name} sank the fleet of {loser.name} with {shots} shots.", (cx, 430), center=True)
        self.text("Press H to see the AI's last probability map.", (cx, 465), "small", theme.TEXT_DIM, True)
        self.result_button.draw(self.surface, self.fonts, mouse)

    # ------------------------------------------------------------------ loop
    def run(self, max_frames: int | None = None) -> None:
        frames = 0
        while self.running and (max_frames is None or frames < max_frames):
            dt = self.clock.tick(60) / 1000
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(dt)
            self.draw()
            frames += 1
        pygame.quit()


def run() -> None:
    App().run()

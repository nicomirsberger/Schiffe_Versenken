"""Turn logic for a full game and a fast single-sided simulation."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from battleship.game.board import Board, ShotOutcome, ShotResult

if TYPE_CHECKING:
    from battleship.ai.base import Agent


class Player(Enum):
    HUMAN = "human"
    AI = "ai"


class Game:
    """A game of a human against an AI agent.

    The human attacks ``ai_board`` and the AI attacks ``human_board``.
    """

    def __init__(
        self,
        human_board: Board,
        ai_board: Board,
        agent: Agent,
        rng: np.random.Generator | None = None,
        starting_player: Player = Player.HUMAN,
    ) -> None:
        if human_board.config != ai_board.config:
            raise ValueError("Both boards must use the same configuration.")
        self.config = human_board.config
        self.human_board = human_board
        self.ai_board = ai_board
        self.agent = agent
        self.rng = rng if rng is not None else np.random.default_rng()
        self.current = starting_player
        self.winner: Player | None = None
        self.history: list[tuple[Player, ShotOutcome]] = []

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    def human_fire(self, coord: tuple[int, int]) -> ShotOutcome:
        self._check_turn(Player.HUMAN)
        outcome = self.ai_board.fire(coord)
        self._after_shot(Player.HUMAN, outcome, self.ai_board)
        return outcome

    def ai_fire(self) -> ShotOutcome:
        self._check_turn(Player.AI)
        coord = self.agent.choose_shot(self.human_board.observation(), self.rng)
        outcome = self.human_board.fire(coord)
        self._after_shot(Player.AI, outcome, self.human_board)
        return outcome

    def _check_turn(self, player: Player) -> None:
        if self.is_over:
            raise RuntimeError("The game is already over.")
        if self.current is not player:
            raise RuntimeError(f"It is not the turn of {player.value}.")

    def _after_shot(self, shooter: Player, outcome: ShotOutcome, target: Board) -> None:
        self.history.append((shooter, outcome))
        if target.all_sunk:
            self.winner = shooter
            return
        if outcome.result is ShotResult.MISS or not self.config.extra_shot_on_hit:
            self.current = Player.AI if shooter is Player.HUMAN else Player.HUMAN


def simulate_attack(agent: Agent, board: Board, rng: np.random.Generator) -> int:
    """Let ``agent`` shoot at ``board`` until every ship is sunk.

    Returns the number of shots needed - the main performance metric
    (lower is better).
    """
    limit = board.config.size**2
    while not board.all_sunk:
        if board.shots_fired >= limit:
            raise RuntimeError("Agent exceeded the maximum number of shots.")
        board.fire(agent.choose_shot(board.observation(), rng))
    return board.shots_fired

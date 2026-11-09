"""Generate a labelled dataset from simulated games (self-play).

Every sample is a snapshot of an ongoing game:

* ``X`` - the encoded observation of the attacker (see ``features.py``)
* ``Y`` - the true ship occupancy of the hidden board (one 0/1 label per cell)

The games are played by a mix of "behaviour" agents so that the dataset
contains realistic game situations (hunting, targeting, late game ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from battleship.ai.base import Agent
from battleship.ai.density import DensityAgent
from battleship.ai.heuristic import HuntTargetAgent, RandomAgent
from battleship.config import DEFAULT_CONFIG, GameConfig
from battleship.data.features import encode_observation
from battleship.game.board import Board, CellState
from battleship.game.placement import random_fleet

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DATASET = DATA_DIR / "processed" / "dataset.npz"

#: Default mix of behaviour agents (name -> share of games).
DEFAULT_BEHAVIOUR_MIX: dict[str, float] = {"random": 0.2, "hunt_target": 0.4, "density": 0.4}


@dataclass
class Dataset:
    X: np.ndarray  # (n_samples, n_features) float32
    Y: np.ndarray  # (n_samples, size*size) uint8
    unknown: np.ndarray  # (n_samples, size*size) bool - cell was unknown in the snapshot
    samples: pd.DataFrame  # one row per sample (game_id, behaviour, turn, ...)
    games: pd.DataFrame  # one row per game (game_id, behaviour, shots, ...)

    def save(self, path: Path = DEFAULT_DATASET) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=self.X, Y=self.Y, unknown=self.unknown)
        self.samples.to_csv(path.with_name(path.stem + "_samples.csv"), index=False)
        self.games.to_csv(path.with_name(path.stem + "_games.csv"), index=False)
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_DATASET) -> Dataset:
        path = Path(path)
        arrays = np.load(path)
        samples = pd.read_csv(path.with_name(path.stem + "_samples.csv"))
        games = pd.read_csv(path.with_name(path.stem + "_games.csv"))
        return cls(arrays["X"], arrays["Y"], arrays["unknown"], samples, games)


def make_behaviour_agents(model_path: Path | None = None) -> dict[str, Agent]:
    agents: dict[str, Agent] = {
        "random": RandomAgent(),
        "hunt_target": HuntTargetAgent(),
        "density": DensityAgent(),
    }
    if model_path is not None:
        from battleship.ai.neural import NeuralAgent

        agents["neural"] = NeuralAgent.from_file(model_path, use_search=False)
        agents["neural_search"] = NeuralAgent.from_file(model_path, use_search=True)
    return agents


def generate_dataset(
    n_games: int,
    config: GameConfig = DEFAULT_CONFIG,
    seed: int = 0,
    snapshot_prob: float = 0.3,
    behaviour_mix: dict[str, float] | None = None,
    model_path: Path | None = None,
    progress: bool = False,
) -> Dataset:
    """Simulate ``n_games`` games and record random snapshots of each."""
    mix = behaviour_mix or DEFAULT_BEHAVIOUR_MIX
    agents = make_behaviour_agents(model_path)
    unknown_names = set(mix) - set(agents)
    if unknown_names:
        raise ValueError(f"Unknown behaviour agents: {unknown_names}")
    names = list(mix)
    shares = np.array([mix[n] for n in names], dtype=float)
    shares /= shares.sum()

    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    unknowns: list[np.ndarray] = []
    sample_rows: list[dict] = []
    game_rows: list[dict] = []

    for game_id in range(n_games):
        behaviour = names[rng.choice(len(names), p=shares)]
        agent = agents[behaviour]
        board = Board(random_fleet(config, rng), config)
        label = board.occupancy().reshape(-1).astype(np.uint8)
        turn = 0
        while not board.all_sunk:
            obs = board.observation()
            if rng.random() < snapshot_prob:
                xs.append(encode_observation(obs))
                ys.append(label)
                unknowns.append(obs.unknown_mask.reshape(-1))
                sample_rows.append(
                    {
                        "game_id": game_id,
                        "behaviour": behaviour,
                        "turn": turn,
                        "open_hits": int(np.count_nonzero(obs.grid == CellState.HIT)),
                        "ships_remaining": len(obs.remaining),
                        "unknown_cells": int(obs.unknown_mask.sum()),
                        "ship_cells_unknown": int((label.astype(bool) & obs.unknown_mask.reshape(-1)).sum()),
                    }
                )
            board.fire(agent.choose_shot(obs, rng))
            turn += 1
        game_rows.append({"game_id": game_id, "behaviour": behaviour, "shots": board.shots_fired})
        if progress and (game_id + 1) % max(1, n_games // 10) == 0:
            print(f"  {game_id + 1}/{n_games} games simulated")

    return Dataset(
        X=np.stack(xs).astype(np.float32),
        Y=np.stack(ys).astype(np.uint8),
        unknown=np.stack(unknowns).astype(bool),
        samples=pd.DataFrame(sample_rows),
        games=pd.DataFrame(game_rows),
    )


def merge_datasets(a: Dataset, b: Dataset) -> Dataset:
    """Concatenate two datasets (game ids of ``b`` are shifted)."""
    offset = int(a.games["game_id"].max()) + 1
    samples_b = b.samples.assign(game_id=b.samples["game_id"] + offset)
    games_b = b.games.assign(game_id=b.games["game_id"] + offset)
    return Dataset(
        np.concatenate([a.X, b.X]),
        np.concatenate([a.Y, b.Y]),
        np.concatenate([a.unknown, b.unknown]),
        pd.concat([a.samples, samples_b], ignore_index=True),
        pd.concat([a.games, games_b], ignore_index=True),
    )

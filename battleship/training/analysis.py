"""Data understanding: descriptive statistics and figures of the dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from battleship.config import DEFAULT_CONFIG, GameConfig
from battleship.data.generate import Dataset
from battleship.game.board import Board
from battleship.game.placement import random_fleet

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def ship_frequency(config: GameConfig = DEFAULT_CONFIG, n_boards: int = 5000, seed: int = 0) -> np.ndarray:
    """How often each cell is occupied under random placement (prior)."""
    rng = np.random.default_rng(seed)
    total = np.zeros((config.size, config.size))
    for _ in range(n_boards):
        total += Board(random_fleet(config, rng), config).occupancy()
    return total / n_boards


def describe(dataset: Dataset) -> dict:
    s, g = dataset.samples, dataset.games
    return {
        "games": int(len(g)),
        "samples": int(len(s)),
        "features": int(dataset.X.shape[1]),
        "labels_per_sample": int(dataset.Y.shape[1]),
        "games_per_behaviour": g["behaviour"].value_counts().to_dict(),
        "mean_shots_per_behaviour": g.groupby("behaviour")["shots"].mean().round(2).to_dict(),
        "share_ship_cells_among_unknown": float(dataset.Y[dataset.unknown].mean()),
        "share_samples_with_open_hit": float((s["open_hits"] > 0).mean()),
    }


def plot_data_understanding(
    dataset: Dataset, config: GameConfig = DEFAULT_CONFIG, out: Path = FIGURES_DIR
) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    freq = ship_frequency(config)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(freq, cmap="viridis")
    ax.set_title("Probability of a ship per cell\n(random placement, no-touch rule)")
    ax.set_xticks(range(config.size), [chr(65 + i) for i in range(config.size)])
    ax.set_yticks(range(config.size), range(1, config.size + 1))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    paths.append(out / "ship_frequency.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for behaviour, group in dataset.games.groupby("behaviour"):
        ax.hist(group["shots"], bins=range(20, 101, 2), alpha=0.6, label=behaviour)
    ax.set_xlabel("Shots needed to sink the whole fleet")
    ax.set_ylabel("Games")
    ax.set_title("Game length by behaviour agent")
    ax.legend()
    fig.tight_layout()
    paths.append(out / "shots_by_behaviour.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    s = dataset.samples
    fig, ax = plt.subplots(figsize=(6, 4))
    by_turn = s.assign(ratio=s["ship_cells_unknown"] / s["unknown_cells"]).groupby("turn")["ratio"].mean()
    ax.plot(by_turn.index, by_turn.values)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Share of unknown cells that hold a ship")
    ax.set_title("Class balance over the course of a game")
    fig.tight_layout()
    paths.append(out / "class_balance.png")
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths

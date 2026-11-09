"""Evaluation of the model (offline metrics) and of the agents (games)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from battleship.ai.base import Agent
from battleship.ai.density import placement_density
from battleship.ai.model import CellProbabilityModel
from battleship.config import DEFAULT_CONFIG, GameConfig
from battleship.data.features import decode_observation
from battleship.data.generate import Dataset
from battleship.game.board import Board
from battleship.game.match import simulate_attack
from battleship.game.placement import random_fleet
from battleship.training.analysis import FIGURES_DIR
from battleship.training.metrics import (
    cell_metrics,
    density_to_probability,
)
from battleship.training.train import split_by_game


def compare_cell_models(
    dataset: Dataset, model: CellProbabilityModel, config: GameConfig = DEFAULT_CONFIG, seed: int = 0
) -> pd.DataFrame:
    """Compare the neural network with two baselines on the held-out test games.

    * prior   - per-cell ship frequency of the training data (ignores the game state)
    * density - constraint search, rescaled to probabilities
    * neural  - the trained network
    * neural + search - product of network and density (what the strong AI uses)
    """
    train, _, test = split_by_game(dataset, seed=seed)
    Y, unknown = dataset.Y[test], dataset.unknown[test]
    X = dataset.X[test]

    prior = np.tile(dataset.Y[train].mean(axis=0), (len(Y), 1))
    expected = np.array([(Y[i] & unknown[i]).sum() for i in range(len(Y))], dtype=float)
    density = np.stack([placement_density(decode_observation(x, config)).reshape(-1) for x in X])
    neural = model.predict_proba(X)
    combined = neural * density / np.maximum(density.max(axis=1, keepdims=True), 1e-9)

    rows = {
        "prior": cell_metrics(prior, Y, unknown),
        "density (search)": cell_metrics(density_to_probability(density, expected), Y, unknown),
        "neural": cell_metrics(neural, Y, unknown),
        "neural + search": cell_metrics(density_to_probability(combined, expected), Y, unknown),
    }
    # Calibration plot of the network.
    frac, mean_pred = calibration_curve(Y[unknown], neural[unknown], n_bins=10)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(mean_pred, frac, "o-", label="neural network")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed share of ship cells")
    ax.set_title("Calibration (test games)")
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=120)
    plt.close(fig)
    return pd.DataFrame(rows).T.round(4)


def evaluate_agents(
    agents: list[Agent], n_games: int = 300, config: GameConfig = DEFAULT_CONFIG, seed: int = 1234
) -> pd.DataFrame:
    """Let every agent play against the *same* boards; returns one row per game."""
    rows = []
    for game in range(n_games):
        fleet = random_fleet(config, np.random.default_rng([seed, game]))
        for agent in agents:
            shots = simulate_attack(agent, Board(fleet, config), np.random.default_rng([seed, game, 1]))
            rows.append({"game": game, "agent": agent.name, "shots": shots})
    return pd.DataFrame(rows)


def summarise_games(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("agent")["shots"].describe()[["mean", "std", "min", "50%", "max"]]
    summary = summary.rename(columns={"50%": "median"}).sort_values("mean")
    return summary.round(2)


def head_to_head(results: pd.DataFrame, a: str, b: str) -> dict[str, float]:
    """Share of boards on which agent ``a`` needed fewer shots than ``b``."""
    pivot = results.pivot(index="game", columns="agent", values="shots")
    return {
        f"{a} better": float((pivot[a] < pivot[b]).mean()),
        "draw": float((pivot[a] == pivot[b]).mean()),
        f"{b} better": float((pivot[a] > pivot[b]).mean()),
    }


def plot_agent_results(results: pd.DataFrame, out: Path = FIGURES_DIR) -> Path:
    order = results.groupby("agent")["shots"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([results.loc[results.agent == a, "shots"] for a in order], tick_labels=order)
    ax.set_ylabel("Shots to sink the whole fleet (lower is better)")
    ax.set_title(f"Agent comparison on {results.game.nunique()} identical boards")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "agent_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_training(report: dict, out: Path = FIGURES_DIR) -> Path:
    fig, ax1 = plt.subplots(figsize=(6, 4))
    epochs = [h["epoch"] for h in report["history"]]
    ax1.plot(epochs, [h["train_loss"] for h in report["history"]], label="training loss (all cells)")
    ax1.plot(epochs, [h["val_loss"] for h in report["history"]], label="validation loss (unknown cells)")
    ax1.axvline(report["best_epoch"], color="gray", linestyle="--", label="best epoch")
    ax1.legend()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (binary cross-entropy)")
    ax1.set_title("Training progress")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "training_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

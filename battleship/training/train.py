"""Data preparation (train/validation/test split) and model training."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from battleship.ai.model import (
    DEFAULT_MODEL_PATH,
    CellProbabilityModel,
    ModelHyperparameters,
)
from battleship.config import DEFAULT_CONFIG, GameConfig
from battleship.data.generate import Dataset
from battleship.training.metrics import cell_metrics


def split_by_game(
    dataset: Dataset, shares: tuple[float, float] = (0.15, 0.15), seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean sample masks (train, validation, test).

    The split is done per *game*, not per snapshot: snapshots of the same game
    share the same hidden board, so splitting them would leak the label into
    the validation/test data.
    """
    rng = np.random.default_rng(seed)
    game_ids = rng.permutation(dataset.games["game_id"].to_numpy())
    n_val, n_test = (int(len(game_ids) * s) for s in shares)
    val_games, test_games = set(game_ids[:n_val].tolist()), set(game_ids[n_val : n_val + n_test].tolist())
    ids = dataset.samples["game_id"]
    is_val, is_test = ids.isin(val_games).to_numpy(), ids.isin(test_games).to_numpy()
    return ~(is_val | is_test), is_val, is_test


def train_model(
    dataset: Dataset,
    config: GameConfig = DEFAULT_CONFIG,
    params: ModelHyperparameters | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[CellProbabilityModel, dict]:
    train, val, test = split_by_game(dataset, seed=seed)
    model = CellProbabilityModel(config, params or ModelHyperparameters())

    start = time.time()
    history = model.fit(
        dataset.X[train],
        dataset.Y[train],
        dataset.X[val],
        dataset.Y[val],
        dataset.unknown[val],
        verbose=verbose,
    )
    duration = time.time() - start

    probs = model.predict_proba(dataset.X[test])
    report = {
        "samples": {"train": int(train.sum()), "validation": int(val.sum()), "test": int(test.sum())},
        "games": {
            name: int(dataset.samples.loc[mask, "game_id"].nunique())
            for name, mask in (("train", train), ("validation", val), ("test", test))
        },
        "hyperparameters": asdict(model.params),
        "epochs_trained": len(history),
        "best_epoch": int(min(history, key=lambda h: h["val_loss"])["epoch"]),
        "training_seconds": round(duration, 1),
        "history": history,
        "test_metrics": cell_metrics(probs, dataset.Y[test], dataset.unknown[test]),
    }
    model.save(model_path)
    if verbose:
        print(f"Trained {len(history)} epochs in {duration:.0f}s (best epoch: {report['best_epoch']})")
        print("Test metrics:", json.dumps(report["test_metrics"], indent=2))
    return model, report

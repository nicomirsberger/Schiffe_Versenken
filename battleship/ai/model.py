"""The learned model: predicts for every cell the probability of a ship.

It is a multi-label classifier (one sigmoid output per cell) implemented as a
multi-layer perceptron with scikit-learn.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

from battleship.config import GameConfig
from battleship.data.features import FEATURE_VERSION, encode_observation, feature_size
from battleship.game.board import Observation

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "cell_model.joblib"


@dataclass
class ModelHyperparameters:
    hidden_layers: tuple[int, ...] = (128,)
    learning_rate: float = 1e-3
    alpha: float = 1e-3  # L2 regularisation
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 4  # stop after this many epochs without improvement
    seed: int = 42


@dataclass
class CellProbabilityModel:
    """Wrapper around the classifier that also stores the game config."""

    config: GameConfig
    params: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    classifier: MLPClassifier | None = None
    feature_version: int = FEATURE_VERSION

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        mask_val: np.ndarray,
        verbose: bool = False,
    ) -> list[dict[str, float]]:
        """Train epoch by epoch and keep the epoch with the best validation loss.

        The validation loss is the binary cross-entropy on the cells that were
        still *unknown* - exactly the cells the agent has to decide about.
        Returns the training history (one dict per epoch).
        """
        if X.shape[1] != feature_size(self.config):
            raise ValueError("Feature size does not match the game configuration.")
        p = self.params
        clf = MLPClassifier(
            hidden_layer_sizes=p.hidden_layers,
            learning_rate_init=p.learning_rate,
            alpha=p.alpha,
            batch_size=p.batch_size,
            random_state=p.seed,
        )
        rng = np.random.default_rng(p.seed)
        history: list[dict[str, float]] = []
        best_loss, best_clf, bad_epochs = np.inf, None, 0
        for epoch in range(1, p.max_epochs + 1):
            order = rng.permutation(len(X))
            if epoch == 1:
                clf.partial_fit(X[order], Y[order], classes=np.arange(Y.shape[1]))
            else:
                clf.partial_fit(X[order], Y[order])
            probs = np.clip(np.asarray(clf.predict_proba(X_val))[mask_val], 1e-6, 1 - 1e-6)
            val_loss = float(log_loss(Y_val[mask_val], probs))
            history.append(
                {"epoch": epoch, "train_loss": float(clf.loss_) / Y.shape[1], "val_loss": val_loss}
            )
            if verbose:
                print(f"  epoch {epoch:2d}  train loss {clf.loss_ / Y.shape[1]:.4f}  val loss {val_loss:.4f}")
            if val_loss < best_loss - 1e-4:
                best_loss, best_clf, bad_epochs = val_loss, copy.deepcopy(clf), 0
            else:
                bad_epochs += 1
                if bad_epochs >= p.patience:
                    break
        self.classifier = best_clf
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilities of shape ``(n_samples, size*size)``."""
        if self.classifier is None:
            raise RuntimeError("The model has not been trained yet.")
        return np.asarray(self.classifier.predict_proba(X))

    def predict_observation(self, obs: Observation) -> np.ndarray:
        """Probability map of shape ``(size, size)`` for one observation."""
        probs = self.predict_proba(encode_observation(obs)[None, :])[0]
        return probs.reshape(self.config.size, self.config.size)

    # ----------------------------------------------------------- persistence
    def save(self, path: Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": asdict(self.config),
                "params": asdict(self.params),
                "feature_version": self.feature_version,
                "classifier": self.classifier,
            },
            path,
            compress=3,
        )
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> CellProbabilityModel:
        data = joblib.load(Path(path))
        if data["feature_version"] != FEATURE_VERSION:
            raise ValueError("Model was trained with an incompatible feature version - retrain it.")
        config = GameConfig(**{**data["config"], "fleet": tuple(data["config"]["fleet"])})
        params = ModelHyperparameters(
            **{**data["params"], "hidden_layers": tuple(data["params"]["hidden_layers"])}
        )
        return cls(config=config, params=params, classifier=data["classifier"])

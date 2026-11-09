import numpy as np

from battleship.data.features import decode_observation, encode_observation, feature_size
from battleship.data.generate import Dataset, generate_dataset, merge_datasets
from battleship.game import Board, random_fleet
from battleship.training.metrics import cell_metrics
from battleship.training.train import split_by_game


def test_feature_roundtrip(small_config, rng):
    board = Board(random_fleet(small_config, rng), small_config)
    for cell in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        board.fire(cell)
    obs = board.observation()
    features = encode_observation(obs)
    assert features.shape == (feature_size(small_config),)
    decoded = decode_observation(features, small_config)
    assert (decoded.grid == obs.grid).all()
    assert decoded.remaining == obs.remaining


def test_generate_and_save(tmp_path, small_config):
    data = generate_dataset(20, small_config, seed=0, snapshot_prob=0.5)
    assert len(data.X) == len(data.Y) == len(data.samples) == len(data.unknown)
    assert data.Y.shape[1] == small_config.size**2
    assert set(data.games["behaviour"]) <= {"random", "hunt_target", "density"}
    path = data.save(tmp_path / "ds.npz")
    loaded = Dataset.load(path)
    np.testing.assert_array_equal(loaded.X, data.X)


def test_split_has_no_game_leakage(small_config):
    data = generate_dataset(60, small_config, seed=0)
    train, val, test = split_by_game(data)
    ids = data.samples["game_id"]
    assert not (train & val).any() and not (train & test).any()
    assert set(ids[train]).isdisjoint(ids[test])
    assert set(ids[val]).isdisjoint(ids[test])


def test_merge(small_config):
    a = generate_dataset(5, small_config, seed=0)
    b = generate_dataset(5, small_config, seed=1)
    merged = merge_datasets(a, b)
    assert len(merged.X) == len(a.X) + len(b.X)
    assert merged.games["game_id"].is_unique


def test_cell_metrics_perfect_prediction():
    Y = np.array([[1, 0, 0, 1], [0, 1, 0, 0]], dtype=np.uint8)
    unknown = np.ones_like(Y, dtype=bool)
    metrics = cell_metrics(Y.astype(float), Y, unknown)
    assert metrics["roc_auc"] == 1.0
    assert metrics["next_shot_hit_rate"] == 1.0

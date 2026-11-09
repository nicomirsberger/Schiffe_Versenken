import numpy as np
import pytest

from battleship.ai import DensityAgent, HuntTargetAgent, NeuralAgent, RandomAgent
from battleship.ai.factory import DIFFICULTIES, create_agent
from battleship.ai.model import CellProbabilityModel, ModelHyperparameters
from battleship.data.generate import generate_dataset
from battleship.game import Board, random_fleet, simulate_attack


@pytest.mark.parametrize("agent_cls", [RandomAgent, HuntTargetAgent, DensityAgent])
def test_agents_finish_game_without_repeating(agent_cls, small_config, rng):
    board = Board(random_fleet(small_config, rng), small_config)
    shots = simulate_attack(agent_cls(), board, rng)
    assert board.all_sunk
    assert sum(small_config.fleet) <= shots <= small_config.size**2


def test_smart_agents_beat_random(small_config):
    def mean_shots(agent):
        rng = np.random.default_rng(0)
        return np.mean(
            [
                simulate_attack(agent, Board(random_fleet(small_config, rng), small_config), rng)
                for _ in range(30)
            ]
        )

    assert mean_shots(DensityAgent()) < mean_shots(RandomAgent())
    assert mean_shots(HuntTargetAgent()) < mean_shots(RandomAgent())


@pytest.fixture
def tiny_model(small_config):
    data = generate_dataset(40, small_config, seed=1, snapshot_prob=0.5)
    model = CellProbabilityModel(small_config, ModelHyperparameters(hidden_layers=(16,), max_epochs=3))
    model.fit(data.X, data.Y, data.X, data.Y, data.unknown)
    return model


@pytest.mark.parametrize("use_search", [False, True])
def test_neural_agent_plays(tiny_model, small_config, rng, use_search):
    board = Board(random_fleet(small_config, rng), small_config)
    agent = NeuralAgent(tiny_model, use_search=use_search)
    simulate_attack(agent, board, rng)
    assert board.all_sunk
    assert agent.last_scores is not None


def test_model_roundtrip(tiny_model, tmp_path, small_config, rng):
    path = tiny_model.save(tmp_path / "model.joblib")
    loaded = CellProbabilityModel.load(path)
    obs = Board(random_fleet(small_config, rng), small_config).observation()
    np.testing.assert_allclose(loaded.predict_observation(obs), tiny_model.predict_observation(obs))


def test_neural_agent_rejects_other_config(tiny_model, rng):
    from battleship.config import DEFAULT_CONFIG

    obs = Board(random_fleet(DEFAULT_CONFIG, rng)).observation()
    with pytest.raises(ValueError):
        NeuralAgent(tiny_model).choose_shot(obs, rng)


def test_factory_creates_every_level(tmp_path):
    for difficulty in DIFFICULTIES:
        agent, _ = create_agent(difficulty.key)
        assert agent is not None
    # Missing model -> fallback with a warning.
    agent, warning = create_agent("hard", model_path=tmp_path / "missing.joblib")
    assert isinstance(agent, DensityAgent) and warning

# Battleship AI

[![CI](https://github.com/nicomirsberger/Schiffe_Versenken/actions/workflows/ci.yml/badge.svg)](https://github.com/nicomirsberger/Schiffe_Versenken/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Play Battleship against an AI opponent. The strongest level combines a
**neural network trained on 200,000 simulated game situations** with a
**search over every possible ship position**. The AI was developed along the
**CRISP-DM** process model.

![Battle screen](docs/screenshots/3_battle_heatmap.png)

## Features

- Battleship with classic rules: 10×10 board, ten ships (1×5, 2×4, 3×3, 4×2),
  ships may not touch, a hit earns another shot
- Five nations (Germany, USA, Japan, United Kingdom, France), each with its own
  battleship silhouette and flag. The ships burn as their fleet is sunk and sink
  when the game is lost.
- Four AI levels, from random shots to neural network + search
- Shot logs with the last three shots of each side
- **AI heatmap:** press `H` to see where the AI expects your ships

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/nicomirsberger/Schiffe_Versenken.git
cd Schiffe_Versenken
pip install -r requirements.txt
python main.py
```

The trained model is included, so no training is needed.

**Controls:** pick both nations and a level in the menu (or press `1`–`4`).
Place ships with a left click, `R` rotates, `Space` places the fleet at random,
`Enter` starts. Click the enemy waters to fire. `H` shows the AI heatmap,
`Esc` returns to the menu.

<p align="center">
  <img src="docs/screenshots/1_menu.png" width="49%" alt="Menu">
  <img src="docs/screenshots/2_setup.png" width="49%" alt="Fleet setup">
</p>

## The four AI levels

The AI cannot see your ships, so every level answers one question: *which
unknown cell most likely holds a ship?* Each level answers it better.

| Level | How it chooses a shot | Learns from data | Avg. shots to win* |
|---|---|:---:|---:|
| Beginner | Random cell | – | 97.7 |
| Easy | Hunt & target: checkerboard search, then attacks around hits | – | 51.6 |
| Medium | Neural network rates every cell | ✓ | 50.2 |
| Hard | Neural network × search over all legal ship positions | ✓ | **47.8** |

\* Shots to sink the whole fleet, measured on the same 300 boards (fewer is
better, 30 is perfect).

- **Search** ([`game/knowledge.py`](battleship/game/knowledge.py)): tries every
  position of every remaining ship, drops those that break a rule and counts how
  many cover each cell.
- **Neural network** ([`ai/model.py`](battleship/ai/model.py)): a multi-layer
  perceptron (605 inputs → 128 → 100 outputs) trained on 12,000 simulated games.
  It learned patterns the rules do not state, such as ships being more common at
  the edges.
- **Hard** ([`ai/neural.py`](battleship/ai/neural.py)) multiplies both: impossible
  cells score zero, and among the possible ones the network decides.

## AI development with CRISP-DM

| Phase | In this project |
|---|---|
| Business understanding | Goal: beat the human hunt & target strategy (51.6 shots) |
| Data understanding | No dataset exists → 12,000 simulated games, ~200,000 snapshots |
| Data preparation | 605 features per board, split **by game** to avoid leakage |
| Modeling | Multi-label MLP, hyperparameter comparison, own early stopping |
| Evaluation | Offline metrics vs. baselines **and** real games on identical boards |
| Deployment | Desktop game with four levels and an explainable heatmap |

The cycle ran **three times**: feature engineering and self-play data each
came from a finding in the evaluation. Details: [docs/CRISP-DM.md](docs/CRISP-DM.md).

![Agent comparison](reports/figures/agent_comparison.png)

## Project structure

```
battleship/
├── game/       # rules only: board, placement, search, turns
├── data/       # feature encoding, dataset generation
├── ai/         # the four agents and the neural network
├── training/   # training, metrics, evaluation
└── ui/         # pygame interface, nations, ship art
models/         # trained model
reports/        # metrics and figures
docs/           # CRISP-DM, architecture, screenshots
tests/          # 59 pytest tests
```

Dependencies point one way (`ui → ai → data → game`), and all agents share one
interface. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Reproduce and develop

```bash
python -m battleship pipeline --games 8000            # generate, analyze, train, evaluate
python -m battleship generate --iterate --games 4000  # add self-play games (iteration 3)
python -m battleship train && python -m battleship evaluate
pip install -r requirements-dev.txt && pytest && ruff check .
```

All random processes are seeded. GitHub Actions runs linting and tests on
every push.

## Authors

Nico Mirsberger & Andre Richter · [MIT License](LICENSE)

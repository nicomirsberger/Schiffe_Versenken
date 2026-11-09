# Quality criteria

This document shows how the project addresses common quality criteria for
software (based on ISO/IEC 25010) and for data science / machine learning projects.

## Software quality

| Criterion | How it is addressed |
|---|---|
| **Functional suitability** | Complete game with rule validation (placement, no-touch rule, turn order, extra shot on hit). Rules are covered by unit tests (`tests/test_board.py`, `test_placement.py`, `test_game.py`). |
| **Reliability** | Invalid actions raise clear errors (shooting twice, shooting outside the board, acting out of turn). If the model is missing, the game falls back to the search agent. |
| **Maintainability: modularity** | Strict layer architecture (`game → data → ai → training/ui`), one responsibility per module, common `Agent` interface ([ARCHITECTURE.md](ARCHITECTURE.md)). |
| **Maintainability: readability** | Type hints throughout, docstrings for every module and public class, consistent formatting with `ruff format`. |
| **Maintainability: changeability** | All rules in one `GameConfig`. New opponents need one class and one line in the factory. |
| **Testability** | 52 automated tests with `pytest`: rules, agents, data pipeline (incl. a leakage test), model saving/loading, UI smoke test. Tests use a small board, so the suite runs in a few seconds. |
| **Usability** | Graphical UI, clear status messages, hover preview while placing ships, a random fleet button, undo, four difficulty levels, on-screen help. |
| **Performance** | ≈ 3 ms per AI decision on the hardest level, vectorised constraint search with NumPy. |
| **Portability** | Pure Python with common packages. Runs on Windows, macOS and Linux. CI tests Python 3.10 and 3.12. |
| **Continuous integration** | GitHub Actions runs linting (`ruff check`, `ruff format --check`) and all tests on every push and pull request. |
| **Licence and openness** | MIT licence, public repository, English documentation. |

## Data science quality

| Criterion | How it is addressed |
|---|---|
| **Process model** | Development along CRISP-DM with three documented iterations ([CRISP-DM.md](CRISP-DM.md)). |
| **Reproducibility** | Whole pipeline as CLI commands. All random generators are seeded. Hyperparameters, metrics and the training history are saved as JSON in `reports/`. |
| **No data leakage** | Train/validation/test split **by game**, verified by an automated test. Hyperparameters are chosen on validation data, and the test set is only used for the final report. |
| **Suitable metrics** | Imbalanced labels (23 % positive) → ROC AUC, log loss, Brier score instead of accuracy. Metrics only on cells that were still unknown. |
| **Baselines** | Every model is compared with a prior, the constraint search, hunt & target and random. |
| **Business evaluation** | Besides offline metrics, every agent plays on the same 300 boards (paired comparison). This revealed a distribution shift that the offline metrics did not show. |
| **Honest reporting** | Insignificant differences are reported as such, and weaknesses are documented (calibration, random placement). |
| **Model versioning** | The model file stores the game configuration and feature version. Incompatible models are rejected. |
| **Explainability** | The in-game heatmap shows where the AI expects ships. The network's input includes the interpretable density map. |

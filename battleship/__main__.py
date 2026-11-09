"""Command line interface.

python -m battleship play                 # start the game
python -m battleship pipeline             # run the full CRISP-DM pipeline
python -m battleship generate|analyze|train|evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from battleship.config import DEFAULT_CONFIG

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def cmd_generate(args: argparse.Namespace) -> None:
    from battleship.ai.model import DEFAULT_MODEL_PATH
    from battleship.data.generate import DEFAULT_DATASET, Dataset, generate_dataset, merge_datasets

    if args.iterate:
        # CRISP-DM iteration 3: the model plays itself and its games are added
        # to the existing data, so the next model also learns from the game
        # situations it creates itself (fixes the distribution shift).
        if not (DEFAULT_MODEL_PATH.exists() and DEFAULT_DATASET.exists()):
            raise SystemExit("--iterate needs an existing dataset and model - run the pipeline first.")
        print(f"Simulating {args.games} games played by the current model ...")
        extra = generate_dataset(
            args.games,
            DEFAULT_CONFIG,
            seed=args.seed + 10_000,
            behaviour_mix={"neural": 0.5, "neural_search": 0.5},
            model_path=DEFAULT_MODEL_PATH,
            progress=True,
        )
        dataset = merge_datasets(Dataset.load(DEFAULT_DATASET), extra)
    else:
        print(f"Simulating {args.games} games ...")
        dataset = generate_dataset(args.games, DEFAULT_CONFIG, seed=args.seed, progress=True)
    path = dataset.save(DEFAULT_DATASET)
    print(f"Saved {len(dataset.X)} samples to {path}")


def cmd_analyze(_: argparse.Namespace) -> None:
    from battleship.data.generate import Dataset
    from battleship.training.analysis import describe, plot_data_understanding

    dataset = Dataset.load()
    summary = describe(dataset)
    print(json.dumps(summary, indent=2))
    _write_json(REPORTS_DIR / "data_summary.json", summary)
    for path in plot_data_understanding(dataset):
        print("Figure:", path)


def cmd_train(args: argparse.Namespace) -> None:
    from battleship.ai.model import ModelHyperparameters
    from battleship.data.generate import Dataset
    from battleship.training.evaluate import plot_training
    from battleship.training.train import train_model

    params = ModelHyperparameters(hidden_layers=tuple(args.hidden), max_epochs=args.epochs)
    _, report = train_model(Dataset.load(), DEFAULT_CONFIG, params)
    _write_json(REPORTS_DIR / "training_report.json", report)
    print("Figure:", plot_training(report))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from battleship.ai import DensityAgent, HuntTargetAgent, NeuralAgent, RandomAgent
    from battleship.ai.model import CellProbabilityModel
    from battleship.data.generate import Dataset
    from battleship.training.evaluate import (
        compare_cell_models,
        evaluate_agents,
        head_to_head,
        plot_agent_results,
        summarise_games,
    )

    model = CellProbabilityModel.load()
    print("Offline evaluation on held-out test games ...")
    offline = compare_cell_models(Dataset.load(), model)
    print(offline.to_string())

    print(f"\nPlaying {args.games} games per agent on identical boards ...")
    agents = [
        RandomAgent(),
        HuntTargetAgent(),
        DensityAgent(),
        NeuralAgent(model, use_search=False),
        NeuralAgent(model, use_search=True),
    ]
    results = evaluate_agents(agents, n_games=args.games)
    summary = summarise_games(results)
    print(summary.to_string())
    duel = head_to_head(results, "neural_search", "density")
    print("Head to head:", duel)
    print("Figure:", plot_agent_results(results))

    _write_json(
        REPORTS_DIR / "evaluation.json",
        {
            "offline": offline.to_dict(orient="index"),
            "games": summary.to_dict(orient="index"),
            "head_to_head": duel,
        },
    )
    (REPORTS_DIR / "results.md").write_text(
        "# Evaluation results\n\n## Cell prediction (held-out test games)\n\n"
        + offline.to_markdown()
        + f"\n\n## Shots to sink the fleet ({args.games} identical boards)\n\n"
        + summary.to_markdown()
        + "\n\n## Head to head: neural_search vs. density\n\n"
        + "\n".join(f"- {k}: {v:.1%}" for k, v in duel.items())
        + "\n"
    )


def cmd_pipeline(args: argparse.Namespace) -> None:
    cmd_generate(args)
    cmd_analyze(args)
    cmd_train(args)
    cmd_evaluate(args)


def cmd_play(args: argparse.Namespace) -> None:
    from battleship.ui.app import run

    run()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="battleship", description="Battleship against a data-driven AI.")
    sub = parser.add_subparsers(dest="command")

    def add_generate_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--games", type=int, default=3000, help="number of simulated games")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--iterate",
            action="store_true",
            help="append games played by the current model to the existing dataset",
        )

    def add_train_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--hidden", type=int, nargs="+", default=[128], help="hidden layer sizes")
        p.add_argument("--epochs", type=int, default=40, help="maximum number of epochs")

    p = sub.add_parser("generate", help="simulate games and build the dataset")
    add_generate_args(p)
    p.set_defaults(func=cmd_generate)

    sub.add_parser("analyze", help="describe the dataset and plot figures").set_defaults(func=cmd_analyze)

    p = sub.add_parser("train", help="train the neural network")
    add_train_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate model and agents")
    p.add_argument("--games", type=int, default=300)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("pipeline", help="generate -> analyze -> train -> evaluate")
    add_generate_args(p)
    add_train_args(p)
    p.set_defaults(func=cmd_pipeline)

    sub.add_parser("play", help="play against the AI").set_defaults(func=cmd_play)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args = parser.parse_args(["play"])
    args.func(args)


if __name__ == "__main__":
    main()

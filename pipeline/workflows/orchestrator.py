from __future__ import annotations

import argparse

from pipeline.cli import (
    COMMAND_ANALYZE_BACKBONES,
    COMMAND_BUILD_BACKBONES,
    COMMAND_FORECAST,
)


def run_command(args: argparse.Namespace) -> None:
    if args.command == COMMAND_FORECAST:
        from pipeline.workflows.forecasting import run_forecasting_workflow

        run_forecasting_workflow(
            config_path=args.config,
            dry_run=args.dry_run,
        )
        return

    if args.command == COMMAND_BUILD_BACKBONES:
        from pipeline.workflows.backbone import run_backbone_generation_workflow

        run_backbone_generation_workflow(
            datasets=args.datasets,
            methods=args.methods,
            alpha=args.alpha,
            min_degree=args.min_degree,
        )
        return

    if args.command == COMMAND_ANALYZE_BACKBONES:
        from pipeline.workflows.analysis import run_backbone_analysis_workflow

        run_backbone_analysis_workflow(
            datasets=args.datasets,
            network_names=args.network_names,
            methods=args.methods,
            alpha=args.alpha,
            original_path=args.original_path,
            backbone_paths=args.backbone_paths,
            dataset_label=args.dataset_label,
            top_k=args.top_k,
            robustness_steps=args.robustness_steps,
            random_trials=args.random_trials,
            output_tag=args.output_tag,
        )
        return

    raise ValueError(f"Comando nao suportado: {args.command!r}")

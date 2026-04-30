from __future__ import annotations

import json

from pipeline.config import load_config
from pipeline.workflows.runtime import run_pipeline


def run_forecasting_workflow(*, config_path: str, dry_run: bool) -> None:
    config = load_config(config_path)

    print(
        json.dumps(
            {
                "workflow": "forecast",
                "config_file": str(config.config_file),
                "mode": config.mode,
                "experiment_scope": config.experiment_scope,
                "dataset_names": config.dataset_names,
                "model_names": config.model_names,
                "results_dir": str(config.results_dir),
                "best_configs_file": (
                    str(config.best_configs_file) if config.best_configs_file else None
                ),
                "dry_run": dry_run,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    run_pipeline(config=config, dry_run=dry_run)

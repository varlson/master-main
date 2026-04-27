from __future__ import annotations

import argparse
import json

from refat.config import load_config
from refat.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline refatorado de busca e execucao de modelos.")
    parser.add_argument(
        "--config",
        default="refat/config.search.example.json",
        help="Caminho para o arquivo JSON de configuracao.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida a configuracao e imprime o plano de execucao sem treinar modelos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print(
        json.dumps(
            {
                "config_file": str(config.config_file),
                "mode": config.mode,
                "experiment_scope": config.experiment_scope,
                "dataset_names": config.dataset_names,
                "model_names": config.model_names,
                "results_dir": str(config.results_dir),
                "best_configs_file": str(config.best_configs_file) if config.best_configs_file else None,
                "dry_run": args.dry_run,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    run_pipeline(config=config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()


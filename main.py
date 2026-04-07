from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import os
import traceback
from typing import Any, Callable

import numpy as np
import torch

from shared.loaders import prepare_dataloaders_from_arrays
from shared.MLFlow import (
    DCRNN_grid_search,
    GraphWaveNet_grid_search,
    MTGNN_grid_search,
    DGCRN_grid_search,
    STICformer_grid_search,
    PatchSTG_grid_search,
    set_results_root,
)
from shared.reproducibility import parse_seeds
from shared.resultSumarization import (
    consolidate_experiment_results,
    create_comparison_report,
    export_best_configs_to_json,
)


@dataclass(frozen=True)
class RuntimeConfig:
    config_source: str
    config_file: Path
    experiment: str
    run_original: bool
    run_backbone: bool
    dataset_names: list[str]
    backbone_dataset_names: list[str]
    original_npy_dir: Path
    backbone_npy_dir: Path
    results_dir: Path
    device: str
    seq_len: int
    horizon: int
    batch_size: int
    epochs: int
    seeds: list[int]
    selection_metric: str
    run_dcrnn: bool
    run_graph_wavenet: bool
    run_mtgnn: bool
    run_dgcrn: bool
    run_sticformer: bool
    run_patchstg: bool
    generate_plots: bool
    plots_num_nodes: int
    plots_max_time_points: int
    run_label: str


def _parse_bool(value: Any, key_name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False

    raise ValueError(f"Valor booleano invalido para '{key_name}': {value!r}")


def _parse_int(value: Any, key_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valor inteiro invalido para '{key_name}': {value!r}") from exc


def _parse_names(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(value, list):
        names = []
        for item in value:
            text = str(item).strip()
            if text:
                names.append(text)
        return names

    raise ValueError(f"Lista de nomes invalida: {value!r}")


def _load_json_config(config_file: Path) -> dict[str, Any]:
    if not config_file.exists():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"O arquivo {config_file} deve conter um objeto JSON na raiz.")

    return data


def _get_json_value(config_data: dict[str, Any], key: str) -> Any:
    if key in config_data:
        return config_data[key]

    lower_key = key.lower()
    if lower_key in config_data:
        return config_data[lower_key]

    return None


def _resolve_raw_value(
    *,
    key: str,
    default: Any,
    config_source: str,
    json_config: dict[str, Any],
) -> Any:
    if config_source == "json":
        value_from_json = _get_json_value(json_config, key)
        if value_from_json is not None:
            return value_from_json

    env_value = os.getenv(key)
    if env_value is not None:
        return env_value

    return default


def load_runtime_config() -> RuntimeConfig:
    config_source = os.getenv("CONFIG_SOURCE", "env").strip().lower()
    if config_source not in {"env", "json"}:
        raise ValueError(
            "CONFIG_SOURCE invalido. Use 'env' ou 'json'."
        )

    config_file = Path(os.getenv("CONFIG_FILE", "config.json"))
    json_config: dict[str, Any] = {}
    if config_source == "json":
        json_config = _load_json_config(config_file)

    dataset_names = _parse_names(
        _resolve_raw_value(
            key="DATASET_NAMES",
            default="",
            config_source=config_source,
            json_config=json_config,
        )
    )
    if not dataset_names:
        dataset_names = _parse_names(
            _resolve_raw_value(
                key="DATASET_NAME",
                default="pems-bay",
                config_source=config_source,
                json_config=json_config,
            )
        )

    if not dataset_names:
        raise ValueError("Nenhum dataset informado em DATASET_NAMES ou DATASET_NAME.")

    experiment = str(
        _resolve_raw_value(
            key="EXPERIMENT",
            default="original",
            config_source=config_source,
            json_config=json_config,
        )
    ).strip().lower()
    if experiment not in {"original", "backbone", "both"}:
        raise ValueError("EXPERIMENT invalido. Use 'original', 'backbone' ou 'both'.")

    run_original = experiment in {"original", "both"}
    run_backbone = experiment in {"backbone", "both"}

    backbone_dataset_names = _parse_names(
        _resolve_raw_value(
            key="BACKBONE_DATASET_NAMES",
            default="",
            config_source=config_source,
            json_config=json_config,
        )
    )
    if run_backbone and not backbone_dataset_names:
        backbone_dataset_names = list(dataset_names)

    epochs = _parse_int(
        _resolve_raw_value(
            key="EPOCHS",
            default="5",
            config_source=config_source,
            json_config=json_config,
        ),
        "EPOCHS",
    )

    run_stamp = datetime.now().strftime("%d_%m_%Y-%Hh_%M")
    run_label = f"{run_stamp}-epoch_{epochs}"

    device_default = "cuda" if torch.cuda.is_available() else "cpu"

    return RuntimeConfig(
        config_source=config_source,
        config_file=config_file,
        experiment=experiment,
        run_original=run_original,
        run_backbone=run_backbone,
        dataset_names=dataset_names,
        backbone_dataset_names=backbone_dataset_names,
        original_npy_dir=Path(
            _resolve_raw_value(
                key="ORIGINAL_NPY_DIR",
                default="data/npy",
                config_source=config_source,
                json_config=json_config,
            )
        ),
        backbone_npy_dir=Path(
            _resolve_raw_value(
                key="BACKBONE_NPY_DIR",
                default="data/npy",
                config_source=config_source,
                json_config=json_config,
            )
        ),
        results_dir=Path(
            _resolve_raw_value(
                key="RESULTS_DIR",
                default="results",
                config_source=config_source,
                json_config=json_config,
            )
        ),
        device=str(
            _resolve_raw_value(
                key="DEVICE",
                default=device_default,
                config_source=config_source,
                json_config=json_config,
            )
        ),
        seq_len=_parse_int(
            _resolve_raw_value(
                key="SEQ_LEN",
                default="12",
                config_source=config_source,
                json_config=json_config,
            ),
            "SEQ_LEN",
        ),
        horizon=_parse_int(
            _resolve_raw_value(
                key="HORIZON",
                default="12",
                config_source=config_source,
                json_config=json_config,
            ),
            "HORIZON",
        ),
        batch_size=_parse_int(
            _resolve_raw_value(
                key="BATCH_SIZE",
                default="8",
                config_source=config_source,
                json_config=json_config,
            ),
            "BATCH_SIZE",
        ),
        epochs=epochs,
        seeds=parse_seeds(
            _resolve_raw_value(
                key="SEEDS",
                default="42",
                config_source=config_source,
                json_config=json_config,
            )
        ),
        selection_metric=str(
            _resolve_raw_value(
                key="SELECTION_METRIC",
                default="val_mae",
                config_source=config_source,
                json_config=json_config,
            )
        ).strip().lower(),
        run_dcrnn=_parse_bool(
            _resolve_raw_value(
                key="RUN_DCRNN",
                default="1",
                config_source=config_source,
                json_config=json_config,
            ),
            "RUN_DCRNN",
        ),
        run_graph_wavenet=_parse_bool(
            _resolve_raw_value(
                key="RUN_GRAPH_WAVENET",
                default="1",
                config_source=config_source,
                json_config=json_config,
            ),
            "RUN_GRAPH_WAVENET",
        ),
        run_mtgnn=_parse_bool(
            _resolve_raw_value(
                key="RUN_MTGNN",
                default="1",
                config_source=config_source,
                json_config=json_config,
            ),
            "RUN_MTGNN",
        ),
        run_dgcrn=_parse_bool(
            _resolve_raw_value(
                key="RUN_DGCRN",
                default="1",
                config_source=config_source,
                json_config=json_config,
            ),
            "RUN_DGCRN",
        ),
        run_sticformer=_parse_bool(
            _resolve_raw_value(
                key="RUN_STICFORMER",
                default="1",
                config_source=config_source,
                json_config=json_config,
            ),
            "RUN_STICFORMER",
        ),
        run_patchstg=_parse_bool(
            _resolve_raw_value(
                key="RUN_PATCHSTG",
                default="1",
                config_source=config_source,
                json_config=json_config,
            ),
            "RUN_PATCHSTG",
        ),
        generate_plots=_parse_bool(
            _resolve_raw_value(
                key="GENERATE_PLOTS",
                default="1",
                config_source=config_source,
                json_config=json_config,
            ),
            "GENERATE_PLOTS",
        ),
        plots_num_nodes=_parse_int(
            _resolve_raw_value(
                key="PLOTS_NUM_NODES",
                default="4",
                config_source=config_source,
                json_config=json_config,
            ),
            "PLOTS_NUM_NODES",
        ),
        plots_max_time_points=_parse_int(
            _resolve_raw_value(
                key="PLOTS_MAX_TIME_POINTS",
                default="350",
                config_source=config_source,
                json_config=json_config,
            ),
            "PLOTS_MAX_TIME_POINTS",
        ),
        run_label=run_label,
    )


def available_datasets(npy_dir: Path) -> list[str]:
    datasets = []
    for file in npy_dir.glob("*-h5.npy"):
        dataset = file.name.replace("-h5.npy", "")
        if (npy_dir / f"{dataset}-adj_mx.npy").exists():
            datasets.append(dataset)
    return sorted(set(datasets))


def resolve_dataset_paths(dataset_name: str, npy_dir: Path) -> tuple[Path, Path]:
    base_datasetname = "metr-la" if "metr-la" in dataset_name else "pems-bay"
    
    base_data_path = npy_dir / f"{base_datasetname}-h5.npy"
    
    data_path = npy_dir / f"{dataset_name}-h5.npy"
    adj_path = npy_dir / f"{dataset_name}-adj_mx.npy"
    
    print(f"Procurando arquivos para dataset '{data_path}' ")
    print(f"Procurando arquivos para dataset '{adj_path}' ")
    if not data_path.exists():
        data_path = base_data_path
        
    

    if not data_path.exists() or not adj_path.exists():
        available = ", ".join(available_datasets(npy_dir)) or "(nenhum encontrado)"
        raise FileNotFoundError(
            f"Arquivos do dataset '{dataset_name}' nao encontrados em {npy_dir}. "
            f"Esperado: '{data_path.name}' e '{adj_path.name}'. "
            f"Disponiveis: {available}"
        )

    return data_path, adj_path


def build_param_grids(seq_len: int, horizon: int, epochs: int) -> tuple[dict, dict, dict, dict, dict, dict]:
    # Grid enxuto por padrao para facilitar execucao inicial.
    dcrnn_param_grid = {
        "input_dim": [1],
        "hidden_dim": [64],
        "output_dim": [1],
        "seq_len": [seq_len],
        "horizon": [horizon],
        "k": [2],
        "dropout": [0.1],
        "lr": [1e-3],
        "weight_decay": [1e-4],
        "epochs": [epochs],
        "patience": [5],
        "use_scheduled_sampling": [False],
        "teacher_forcing_ratio": [0.5],
    }

    graph_wavenet_param_grid = {
        "input_dim": [1],
        "hidden_dim": [64],
        "output_dim": [1],
        "seq_len": [seq_len],
        "horizon": [horizon],
        "num_blocks": [3],
        "dilation_base": [2],
        "k": [2],
        "dropout": [0.1],
        "lr": [1e-3],
        "weight_decay": [1e-4],
        "epochs": [epochs],
        "patience": [5],
    }

    mtgnn_param_grid = {
        "input_dim": [1],
        "hidden_dim": [64],
        "output_dim": [1],
        "seq_len": [seq_len],
        "horizon": [horizon],
        "num_blocks": [3],
        "kernel_size": [2],
        "dilation_base": [2],
        "gcn_depth": [2],
        "propalpha": [0.05],
        "node_dim": [16],
        "dropout": [0.1],
        "lr": [1e-3],
        "weight_decay": [1e-4],
        "epochs": [epochs],
        "patience": [5],
    }

    dgcrn_param_grid = {
        "input_dim": [1],
        "hidden_dim": [64],
        "output_dim": [1],
        "seq_len": [seq_len],
        "horizon": [horizon],
        "node_dim": [16],
        "gcn_depth": [2],
        "dropout": [0.1],
        "lr": [1e-3],
        "weight_decay": [1e-4],
        "epochs": [epochs],
        "patience": [5],
    }

    sticformer_param_grid = {
        "input_dim": [1],
        "hidden_dim": [64],
        "output_dim": [1],
        "seq_len": [seq_len],
        "horizon": [horizon],
        "num_layers": [2],
        "num_heads": [4],
        "ff_multiplier": [2],
        "dropout": [0.1],
        "lr": [1e-3],
        "weight_decay": [1e-4],
        "epochs": [epochs],
        "patience": [5],
    }

    patchstg_param_grid = {
        "input_dim": [1],
        "hidden_dim": [64],
        "output_dim": [1],
        "seq_len": [seq_len],
        "horizon": [horizon],
        "patch_len": [4],
        "patch_stride": [2],
        "num_layers": [2],
        "num_heads": [4],
        "ff_multiplier": [2],
        "dropout": [0.1],
        "lr": [1e-3],
        "weight_decay": [1e-4],
        "epochs": [epochs],
        "patience": [5],
    }

    return (
        dcrnn_param_grid,
        graph_wavenet_param_grid,
        mtgnn_param_grid,
        dgcrn_param_grid,
        sticformer_param_grid,
        patchstg_param_grid,
    )


def run_model_experiment(
    *,
    model_name: str,
    grid_search_fn: Callable,
    param_grid: dict,
    train_loader,
    val_loader,
    test_loader,
    adj_mx: np.ndarray,
    num_nodes: int,
    dataset_name: str,
    normalization_stats: dict,
    experiment_type: str,
    config: RuntimeConfig,
) -> dict | None:
    experiment_name = f"{experiment_type}_{dataset_name}_{model_name}_{config.run_label}"

    print(f"\n{'#' * 90}")
    print(f"Iniciando experimento: {experiment_name}")
    print(f"Modelo: {model_name} | Device: {config.device}")
    print(f"{'#' * 90}")

    grid_result = grid_search_fn(
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj_mx=adj_mx,
        num_nodes=num_nodes,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        device=config.device,
        normalization_stats=normalization_stats,
        seeds=config.seeds,
        selection_metric=config.selection_metric,
        generate_plots=config.generate_plots,
        num_nodes_to_plot=config.plots_num_nodes,
        max_time_points=config.plots_max_time_points,
    )

    final_summary = grid_result.get("final_summary") if grid_result else None
    if not final_summary:
        print(f"Sem resultados validos para {model_name}.")
        return None

    return {
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "trial_results": grid_result.get("trial_results", []),
        "config_summaries": grid_result.get("config_summaries", []),
        "selected_config": grid_result.get("selected_config"),
        "final_test_results": grid_result.get("final_test_results", []),
        "final_summary": final_summary,
        "metadata": {
            "run_label": config.run_label,
            "experiment_type": experiment_type,
            "device": config.device,
            "seeds": config.seeds,
            "selection_metric": config.selection_metric,
            "num_nodes": num_nodes,
            "seq_len": config.seq_len,
            "horizon": config.horizon,
            "batch_size": config.batch_size,
            "normalization_stats": normalization_stats,
            "total_trials": len(grid_result.get("trial_results", [])),
            "num_configs_tested": len(grid_result.get("config_summaries", [])),
            "selected_config": grid_result.get("selected_config") or {},
        },
    }


def consolidate_outputs(
    *,
    experiments_data: list[dict],
    output_scope: str,
    results_root: Path,
    run_label: str,
) -> None:
    results_root.mkdir(parents=True, exist_ok=True)

    csv_dir = results_root / "csv"
    json_dir = results_root / "json"
    md_dir = results_root / "md"

    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{output_scope}_{run_label}"
    consolidated_df = consolidate_experiment_results(
        experiments_data=experiments_data,
        output_csv=f"{prefix}_consolidated_experiments.csv",
        output_json=f"{prefix}_consolidated_experiments.json",
        primary_metric="test_mae_mean",
        save_path=results_root,
    )

    if consolidated_df.empty:
        print("Consolidacao vazia. Nenhum relatorio adicional foi gerado.")
        return

    create_comparison_report(
        consolidated_df=consolidated_df,
        output_file=f"{prefix}_comparison_report.md",
        save_path=results_root,
    )

    export_best_configs_to_json(
        consolidated_df=consolidated_df,
        output_file=f"{prefix}_best_configs.json",
        save_path=results_root,
    )

    print("\nArquivos consolidados gerados em:")
    print(f"- {csv_dir / f'{prefix}_consolidated_experiments.csv'}")
    print(f"- {json_dir / f'{prefix}_consolidated_experiments.json'}")
    print(f"- {md_dir / f'{prefix}_comparison_report.md'}")
    print(f"- {json_dir / f'{prefix}_best_configs.json'}")


def run_dataset_pipeline(
    *,
    dataset_name: str,
    npy_dir: Path,
    experiment_type: str,
    results_root: Path,
    config: RuntimeConfig,
) -> list[dict]:
    print(f"\n{'=' * 100}")
    print(f"Tipo de experimento: {experiment_type}")
    print(f"Dataset selecionado: {dataset_name}")
    print(f"Diretorio NPY: {npy_dir}")
    print(f"Device: {config.device}")
    print(f"{'=' * 100}")

    data_path, adj_path = resolve_dataset_paths(dataset_name, npy_dir)
    print(f"Arquivo de serie temporal: {data_path}")
    print(f"Arquivo de adjacencia: {adj_path}")

    data = np.load(data_path)
    adj = np.load(adj_path)

    train_loader, val_loader, test_loader, num_nodes, adj_mx, stats = (
        prepare_dataloaders_from_arrays(
            data=data,
            adj_mx=adj,
            seq_len=config.seq_len,
            horizon=config.horizon,
            batch_size=config.batch_size,
        )
    )

    (
        dcrnn_param_grid,
        graph_wavenet_param_grid,
        mtgnn_param_grid,
        dgcrn_param_grid,
        sticformer_param_grid,
        patchstg_param_grid,
    ) = build_param_grids(config.seq_len, config.horizon, config.epochs)

    model_plan = [
        ("DCRNN", config.run_dcrnn, DCRNN_grid_search, dcrnn_param_grid),
        ("GraphWaveNet", config.run_graph_wavenet, GraphWaveNet_grid_search, graph_wavenet_param_grid),
        ("MTGNN", config.run_mtgnn, MTGNN_grid_search, mtgnn_param_grid),
        ("DGCRN", config.run_dgcrn, DGCRN_grid_search, dgcrn_param_grid),
        ("STICformer", config.run_sticformer, STICformer_grid_search, sticformer_param_grid),
        ("PatchSTG", config.run_patchstg, PatchSTG_grid_search, patchstg_param_grid),
    ]

    experiments_data: list[dict] = []
    for model_name, enabled, grid_search_fn, param_grid in model_plan:
        if not enabled:
            continue

        try:
            result = run_model_experiment(
                model_name=model_name,
                grid_search_fn=grid_search_fn,
                param_grid=param_grid,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
                normalization_stats=stats,
                experiment_type=experiment_type,
                config=config,
            )
            if result is not None:
                experiments_data.append(result)
        except Exception:
            print(f"\nFalha ao executar {model_name} para dataset '{dataset_name}'.")
            traceback.print_exc()

    if not experiments_data:
        print(f"\nNenhum experimento gerou resultados para '{dataset_name}'.")
        return []

    consolidate_outputs(
        experiments_data=experiments_data,
        output_scope=dataset_name,
        results_root=results_root,
        run_label=config.run_label,
    )
    return experiments_data


def run_experiment_group(
    *,
    experiment_type: str,
    dataset_names: list[str],
    npy_dir: Path,
    config: RuntimeConfig,
) -> None:
    results_root = config.results_dir / experiment_type
    set_results_root(results_root)

    print(f"\n{'=' * 100}")
    print(f"Iniciando grupo de experimento: {experiment_type}")
    print(f"Datasets: {dataset_names}")
    print(f"Diretorio de resultados: {results_root}")
    print(f"{'=' * 100}")

    all_experiments_data: list[dict] = []

    for dataset_name in dataset_names:
        try:
            dataset_experiments = run_dataset_pipeline(
                dataset_name=dataset_name,
                npy_dir=npy_dir,
                experiment_type=experiment_type,
                results_root=results_root,
                config=config,
            )
            if dataset_experiments:
                all_experiments_data.extend(dataset_experiments)
        except Exception:
            print(f"\nFalha ao processar dataset '{dataset_name}' no grupo '{experiment_type}'.")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_experiments_data:
        print(f"\nNenhum experimento gerou resultados no grupo '{experiment_type}'.")
        return

    if len(dataset_names) > 1:
        consolidate_outputs(
            experiments_data=all_experiments_data,
            output_scope="all-datasets",
            results_root=results_root,
            run_label=config.run_label,
        )


def main() -> None:
    config = load_runtime_config()

    print("Configuracao de execucao:")
    print(f"- Fonte de configuracao: {config.config_source}")
    if config.config_source == "json":
        print(f"- Arquivo de configuracao: {config.config_file}")
    print(f"- EXPERIMENT: {config.experiment}")
    print(f"- RUN_LABEL: {config.run_label}")
    print(f"- DEVICE: {config.device}")
    print(f"- SEEDS: {config.seeds}")
    print(f"- SELECTION_METRIC: {config.selection_metric}")

    if config.run_original:
        run_experiment_group(
            experiment_type="original",
            dataset_names=config.dataset_names,
            npy_dir=config.original_npy_dir,
            config=config,
        )

    if config.run_backbone:
        run_experiment_group(
            experiment_type="backbone",
            dataset_names=config.backbone_dataset_names,
            npy_dir=config.backbone_npy_dir,
            config=config,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import traceback

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
)
from shared.resultSumarization import (
    consolidate_experiment_results,
    create_comparison_report,
    export_best_configs_to_json,
)


# =========================
# CONFIGURACAO DO PIPELINE
# =========================

NPY_DIR = Path("data/npy")
RESULTS_DIR = Path("results")
RESULTS_CSV_DIR = RESULTS_DIR / "csv"
RESULTS_JSON_DIR = RESULTS_DIR / "json"
RESULTS_MD_DIR = RESULTS_DIR / "md"

DATASET_NAME = os.getenv("DATASET_NAME", "pems-bay")  # ex: "pems-bay" ou "metr-la"
DATASET_NAMES_ENV = os.getenv("DATASET_NAMES", "metr-la")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = int(os.getenv("SEQ_LEN", "12"))
HORIZON = int(os.getenv("HORIZON", "12"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

RUN_DCRNN = os.getenv("RUN_DCRNN", "1") == "1"
EPOCHS = int(os.getenv("EPOCHS", "5"))
RUN_GRAPH_WAVENET = os.getenv("RUN_GRAPH_WAVENET", "1") == "1"
RUN_MTGNN = os.getenv("RUN_MTGNN", "1") == "1"
RUN_DGCRN = os.getenv("RUN_DGCRN", "1") == "1"
RUN_STICFORMER = os.getenv("RUN_STICFORMER", "1") == "1"
RUN_PATCHSTG = os.getenv("RUN_PATCHSTG", "1") == "1"
GENERATE_PLOTS = os.getenv("GENERATE_PLOTS", "1") == "1"
PLOTS_NUM_NODES = int(os.getenv("PLOTS_NUM_NODES", "4"))
PLOTS_MAX_TIME_POINTS = int(os.getenv("PLOTS_MAX_TIME_POINTS", "350"))

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_dataset_names() -> list[str]:
    """
    Prioridade:
    1) DATASET_NAMES="pems-bay,metr-la,..."
    2) DATASET_NAME="pems-bay" (compatibilidade)
    """
    if DATASET_NAMES_ENV.strip():
        names = [name.strip() for name in DATASET_NAMES_ENV.split(",") if name.strip()]
    else:
        names = [DATASET_NAME]

    # remove duplicados preservando ordem
    unique_names = []
    for name in names:
        if name not in unique_names:
            unique_names.append(name)
    return unique_names


def available_datasets(npy_dir: Path) -> list[str]:
    datasets = set()
    for file in npy_dir.glob("*-h5.npy"):
        datasets.add(file.name.replace("-h5.npy", ""))
    return sorted(datasets)


def resolve_dataset_paths(dataset_name: str, npy_dir: Path) -> tuple[Path, Path]:
    data_path = npy_dir / f"{dataset_name}-h5.npy"
    adj_path = npy_dir / f"{dataset_name}-adj_mx.npy"

    if not data_path.exists() or not adj_path.exists():
        available = ", ".join(available_datasets(npy_dir)) or "(nenhum encontrado)"
        raise FileNotFoundError(
            f"Arquivos do dataset '{dataset_name}' nao encontrados em {npy_dir}. "
            f"Esperado: '{data_path.name}' e '{adj_path.name}'. "
            f"Disponiveis: {available}"
        )

    return data_path, adj_path


def build_param_grids(seq_len: int, horizon: int) -> tuple[dict, dict, dict, dict, dict, dict]:
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
        "epochs": [EPOCHS],
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
        "epochs": [EPOCHS],
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
        "epochs": [EPOCHS],
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
        "epochs": [EPOCHS],
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
        "epochs": [EPOCHS],
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
        "epochs": [EPOCHS],
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
    model_name: str,
    grid_search_fn,
    param_grid: dict,
    train_loader,
    val_loader,
    test_loader,
    adj_mx: np.ndarray,
    num_nodes: int,
    dataset_name: str,
    normalization_stats: dict,
) -> dict | None:
    experiment_name = f"{dataset_name}_{model_name}_{RUN_ID}"

    print(f"\n{'#' * 90}")
    print(f"Iniciando experimento: {experiment_name}")
    print(f"Modelo: {model_name} | Device: {DEVICE}")
    print(f"{'#' * 90}")

    all_results, best_result, df_best = grid_search_fn(
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj_mx=adj_mx,
        num_nodes=num_nodes,
        experiment_name=experiment_name,
        device=DEVICE,
        normalization_stats=normalization_stats,
        generate_plots=GENERATE_PLOTS,
        num_nodes_to_plot=PLOTS_NUM_NODES,
        max_time_points=PLOTS_MAX_TIME_POINTS,
    )

    if df_best is None or df_best.empty:
        print(f"Sem resultados validos para {model_name}.")
        return None

    return {
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "df_best": df_best,
        "metadata": {
            "run_id": RUN_ID,
            "device": DEVICE,
            "num_nodes": num_nodes,
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "batch_size": BATCH_SIZE,
            "normalization_stats": normalization_stats,
            "total_configs": len(all_results) if all_results else 0,
            "best_result": best_result if best_result else {},
        },
    }


def consolidate_outputs(experiments_data: list[dict], output_scope: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_MD_DIR.mkdir(parents=True, exist_ok=True)

    prefix = f"{output_scope}_{RUN_ID}"
    consolidated_df = consolidate_experiment_results(
        experiments_data=experiments_data,
        output_csv=f"{prefix}_consolidated_experiments.csv",
        output_json=f"{prefix}_consolidated_experiments.json",
        primary_metric="MAE",
        save_path=RESULTS_DIR,
    )

    if consolidated_df.empty:
        print("Consolidacao vazia. Nenhum relatorio adicional foi gerado.")
        return

    create_comparison_report(
        consolidated_df=consolidated_df,
        output_file=f"{prefix}_comparison_report.md",
        save_path=RESULTS_DIR,
    )

    export_best_configs_to_json(
        consolidated_df=consolidated_df,
        output_file=f"{prefix}_best_configs.json",
        save_path=RESULTS_DIR,
    )

    print("\nArquivos consolidados gerados em:")
    print(f"- {RESULTS_CSV_DIR / f'{prefix}_consolidated_experiments.csv'}")
    print(f"- {RESULTS_JSON_DIR / f'{prefix}_consolidated_experiments.json'}")
    print(f"- {RESULTS_MD_DIR / f'{prefix}_comparison_report.md'}")
    print(f"- {RESULTS_JSON_DIR / f'{prefix}_best_configs.json'}")


def run_dataset_pipeline(dataset_name: str) -> list[dict]:
    print(f"\n{'=' * 100}")
    print(f"Dataset selecionado: {dataset_name}")
    print(f"Diretorio NPY: {NPY_DIR}")
    print(f"Device: {DEVICE}")
    print(f"{'=' * 100}")

    data_path, adj_path = resolve_dataset_paths(dataset_name, NPY_DIR)
    print(f"Arquivo de serie temporal: {data_path}")
    print(f"Arquivo de adjacencia: {adj_path}")

    data = np.load(data_path)
    adj = np.load(adj_path)

    train_loader, val_loader, test_loader, num_nodes, adj_mx, stats = (
        prepare_dataloaders_from_arrays(
            data=data,
            adj_mx=adj,
            seq_len=SEQ_LEN,
            horizon=HORIZON,
            batch_size=BATCH_SIZE,
        )
    )

    (
        dcrnn_param_grid,
        graph_wavenet_param_grid,
        mtgnn_param_grid,
        dgcrn_param_grid,
        sticformer_param_grid,
        patchstg_param_grid,
    ) = build_param_grids(SEQ_LEN, HORIZON)
    experiments_data: list[dict] = []

    if RUN_DCRNN:
        try:
            result = run_model_experiment(
                model_name="DCRNN",
                grid_search_fn=DCRNN_grid_search,
                param_grid=dcrnn_param_grid,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
                normalization_stats=stats,
            )
            if result is not None:
                experiments_data.append(result)
        except Exception:
            print(f"\nFalha ao executar DCRNN para dataset '{dataset_name}'.")
            traceback.print_exc()

    if RUN_GRAPH_WAVENET:
        try:
            result = run_model_experiment(
                model_name="GraphWaveNet",
                grid_search_fn=GraphWaveNet_grid_search,
                param_grid=graph_wavenet_param_grid,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
                normalization_stats=stats,
            )
            if result is not None:
                experiments_data.append(result)
        except Exception:
            print(f"\nFalha ao executar GraphWaveNet para dataset '{dataset_name}'.")
            traceback.print_exc()

    if RUN_MTGNN:
        try:
            result = run_model_experiment(
                model_name="MTGNN",
                grid_search_fn=MTGNN_grid_search,
                param_grid=mtgnn_param_grid,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
                normalization_stats=stats,
            )
            if result is not None:
                experiments_data.append(result)
        except Exception:
            print(f"\nFalha ao executar MTGNN para dataset '{dataset_name}'.")
            traceback.print_exc()

    if RUN_DGCRN:
        try:
            result = run_model_experiment(
                model_name="DGCRN",
                grid_search_fn=DGCRN_grid_search,
                param_grid=dgcrn_param_grid,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
                normalization_stats=stats,
            )
            if result is not None:
                experiments_data.append(result)
        except Exception:
            print(f"\nFalha ao executar DGCRN para dataset '{dataset_name}'.")
            traceback.print_exc()

    if RUN_STICFORMER:
        try:
            result = run_model_experiment(
                model_name="STICformer",
                grid_search_fn=STICformer_grid_search,
                param_grid=sticformer_param_grid,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
                normalization_stats=stats,
            )
            if result is not None:
                experiments_data.append(result)
        except Exception:
            print(f"\nFalha ao executar STICformer para dataset '{dataset_name}'.")
            traceback.print_exc()

    if RUN_PATCHSTG:
        try:
            result = run_model_experiment(
                model_name="PatchSTG",
                grid_search_fn=PatchSTG_grid_search,
                param_grid=patchstg_param_grid,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                dataset_name=dataset_name,
                normalization_stats=stats,
            )
            if result is not None:
                experiments_data.append(result)
        except Exception:
            print(f"\nFalha ao executar PatchSTG para dataset '{dataset_name}'.")
            traceback.print_exc()

    if not experiments_data:
        print(f"\nNenhum experimento gerou resultados para '{dataset_name}'.")
        return []

    consolidate_outputs(experiments_data, dataset_name)
    return experiments_data


def main() -> None:
    dataset_names = parse_dataset_names()
    print(f"Datasets da execucao: {dataset_names}")
    print(f"RUN_ID: {RUN_ID}")

    all_experiments_data: list[dict] = []

    for dataset_name in dataset_names:
        try:
            dataset_experiments = run_dataset_pipeline(dataset_name)
            if dataset_experiments:
                all_experiments_data.extend(dataset_experiments)
        except Exception:
            print(f"\nFalha ao processar dataset '{dataset_name}'. Continuando...")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not all_experiments_data:
        print("\nNenhum experimento gerou resultados em nenhum dataset. Consolidacao global nao executada.")
        return

    if len(dataset_names) > 1:
        consolidate_outputs(all_experiments_data, "all-datasets")


if __name__ == "__main__":
    main()

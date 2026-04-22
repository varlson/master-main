from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from models.MTGNN import MTGNN
from models.STICformer import STICformer
from models.WaveNet import GraphWaveNet
from shared.loaders import prepare_dataloaders_from_arrays
from shared.metrics import (
    REGRESSION_METRICS,
    compute_regression_metrics,
    confidence_interval_95,
    denormalize_arrays,
    prefix_metrics,
    summarize_metric_dicts,
)
from shared.reproducibility import parse_seeds
from shared.resultSumarization import (
    consolidate_experiment_results,
    create_comparison_report,
)
from shared.visualization import generate_model_diagnostics


SUPPORTED_MODELS = ("GraphWaveNet", "MTGNN", "STICformer")
SUPPORTED_BACKBONE_METHODS = ("disp_fil", "nois_corr", "high_sal")


@dataclass(frozen=True)
class SimpleConfig:
    config_file: Path
    params_file: Path
    experiment: str
    run_original: bool
    run_backbone: bool
    dataset_names: list[str]
    backbone_dataset_names: list[str]
    backbone_methods: list[str]
    backbone_alpha: float
    data_dir: Path
    results_dir: Path
    device: str
    seq_len: int
    horizon: int
    batch_size: int
    epochs: int
    seeds: list[int]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    normalize: bool
    normalization_method: str
    generate_plots: bool
    plots_num_nodes: int
    plots_max_time_points: int
    selection_metric: str
    run_label: str


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo JSON nao encontrado: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"O arquivo {path} deve conter um objeto JSON na raiz.")

    return payload


def _resolve_path(raw_value: Any, *, base_dir: Path, default: str) -> Path:
    text = str(raw_value if raw_value is not None else default).strip()
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_dataset_names(raw_value: Any) -> list[str]:
    if raw_value is None:
        return ["pems-bay", "metr-la"]

    if isinstance(raw_value, str):
        names = [item.strip() for item in raw_value.split(",") if item.strip()]
        return names or ["pems-bay", "metr-la"]

    if isinstance(raw_value, list):
        names = [str(item).strip() for item in raw_value if str(item).strip()]
        return names or ["pems-bay", "metr-la"]

    raise ValueError(f"Lista de datasets invalida: {raw_value!r}")


def _parse_name_list(raw_value: Any, *, default: list[str] | tuple[str, ...]) -> list[str]:
    if raw_value is None:
        return list(default)

    if isinstance(raw_value, str):
        names = [item.strip() for item in raw_value.split(",") if item.strip()]
        return names or list(default)

    if isinstance(raw_value, list):
        names = [str(item).strip() for item in raw_value if str(item).strip()]
        return names or list(default)

    raise ValueError(f"Lista de nomes invalida: {raw_value!r}")


def _alpha_cut_name(alpha: float) -> str:
    # Mantem o typo historico usado pelos artefatos ja gerados: "alpah_filter".
    return f"alpah_filter{str(alpha).replace('.', '_')}"


def _backbone_dataset_name(dataset_name: str, method: str, alpha: float) -> str:
    return f"{dataset_name}-by-{method}-with-{_alpha_cut_name(alpha)}"


def _infer_base_dataset_name(dataset_name: str) -> str | None:
    if dataset_name == "metr-la" or dataset_name.startswith("metr-la-"):
        return "metr-la"
    if dataset_name == "pems-bay" or dataset_name.startswith("pems-bay-"):
        return "pems-bay"
    return None


def _method_matches_backbone_name(backbone_name: str, methods: list[str]) -> bool:
    return any(f"-by-{method}-with-" in backbone_name for method in methods)


def _read_backbone_names_file(data_dir: Path) -> list[str]:
    names_file = data_dir / "backbone_data_names.txt"
    if not names_file.exists():
        return []

    with names_file.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def _resolve_backbone_dataset_names(
    *,
    raw_value: Any,
    dataset_names: list[str],
    data_dir: Path,
    methods: list[str],
    alpha: float,
    force_generate: bool,
) -> list[str]:
    explicit_names = _parse_name_list(raw_value, default=[])
    if explicit_names:
        return explicit_names

    if not force_generate:
        discovered_names = []
        for dataset_name in dataset_names:
            prefix = f"{dataset_name}-by-"
            for backbone_name in _read_backbone_names_file(data_dir):
                if backbone_name.startswith(prefix) and _method_matches_backbone_name(
                    backbone_name, methods
                ):
                    discovered_names.append(backbone_name)

        if discovered_names:
            return list(dict.fromkeys(discovered_names))

    generated_names = [
        _backbone_dataset_name(dataset_name, method, alpha)
        for dataset_name in dataset_names
        for method in methods
    ]
    return list(dict.fromkeys(generated_names))


def _resolve_single_seed(raw_value: Any) -> list[int]:
    parsed_seeds = parse_seeds(raw_value)
    if len(parsed_seeds) > 1:
        print(
            "Pipeline simples em modo rapido: usando apenas a primeira seed "
            f"({parsed_seeds[0]}) e ignorando as demais {parsed_seeds[1:]}."
        )
    return [parsed_seeds[0]]


def _resolve_device(requested_device: Any) -> str:
    requested = str(requested_device or "").strip().lower()
    if not requested:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA nao disponivel neste ambiente. Usando CPU no pipeline simples.")
        return "cpu"

    return requested


def _set_fast_single_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        # Mantem inicializacao consistente por seed, mas privilegia throughput.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def load_simple_config(config_file: Path, params_file: Path) -> SimpleConfig:
    payload = _load_json_object(config_file)
    base_dir = config_file.parent

    experiment = str(payload.get("experiment", "original")).strip().lower()
    if experiment not in {"original", "backbone", "both"}:
        raise ValueError("experiment invalido. Use 'original', 'backbone' ou 'both'.")

    dataset_names = _parse_dataset_names(payload.get("dataset_names"))
    data_dir = _resolve_path(payload.get("data_dir"), base_dir=base_dir, default="data/npy")
    backbone_methods = _parse_name_list(
        payload.get("backbone_methods"),
        default=SUPPORTED_BACKBONE_METHODS,
    )
    invalid_methods = sorted(set(backbone_methods) - set(SUPPORTED_BACKBONE_METHODS))
    if invalid_methods:
        raise ValueError(f"Metodos de backbone nao suportados: {invalid_methods}")

    alpha_was_provided = "backbone_alpha" in payload or "alpha" in payload
    backbone_alpha = float(payload.get("backbone_alpha", payload.get("alpha", 0.4)))
    backbone_dataset_names = _resolve_backbone_dataset_names(
        raw_value=payload.get("backbone_dataset_names"),
        dataset_names=dataset_names,
        data_dir=data_dir,
        methods=backbone_methods,
        alpha=backbone_alpha,
        force_generate=alpha_was_provided,
    )

    train_ratio = float(payload.get("train_ratio", 0.7))
    val_ratio = float(payload.get("val_ratio", 0.1))
    test_ratio = float(payload.get("test_ratio", 0.2))
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio devem somar 1.0.")

    normalization_method = str(payload.get("normalization_method", "zscore")).strip().lower()
    if normalization_method not in {"zscore", "minmax"}:
        raise ValueError("normalization_method invalido. Use 'zscore' ou 'minmax'.")

    run_label = str(payload.get("run_label", "")).strip()
    if not run_label:
        run_label = f"{datetime.now().strftime('%d_%m_%Y-%Hh_%M')}-epoch_{int(payload.get('epochs', 5))}"

    return SimpleConfig(
        config_file=config_file.resolve(),
        params_file=params_file.resolve(),
        experiment=experiment,
        run_original=experiment in {"original", "both"},
        run_backbone=experiment in {"backbone", "both"},
        dataset_names=dataset_names,
        backbone_dataset_names=backbone_dataset_names,
        backbone_methods=backbone_methods,
        backbone_alpha=backbone_alpha,
        data_dir=data_dir,
        results_dir=_resolve_path(
            payload.get("results_dir"),
            base_dir=base_dir,
            default="resultsbysimplemain",
        ),
        device=_resolve_device(payload.get("device")),
        seq_len=int(payload.get("seq_len", 12)),
        horizon=int(payload.get("horizon", 12)),
        batch_size=int(payload.get("batch_size", 64)),
        epochs=int(payload.get("epochs", 5)),
        seeds=_resolve_single_seed(payload.get("seed", payload.get("seeds", [42]))),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize=bool(payload.get("normalize", True)),
        normalization_method=normalization_method,
        generate_plots=bool(payload.get("generate_plots", True)),
        plots_num_nodes=int(payload.get("plots_num_nodes", 4)),
        plots_max_time_points=int(payload.get("plots_max_time_points", 350)),
        selection_metric=str(payload.get("selection_metric", "val_mae")).strip().lower(),
        run_label=run_label,
    )


def load_model_params(params_file: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json_object(params_file)
    model_payload = payload.get("models", payload)
    if not isinstance(model_payload, dict):
        raise ValueError(
            f"O arquivo {params_file} deve conter um objeto JSON com os parametros dos modelos."
        )

    unknown_models = sorted(set(model_payload) - set(SUPPORTED_MODELS))
    if unknown_models:
        raise ValueError(f"Modelos nao suportados no pipeline simples: {unknown_models}")

    missing_models = [model_name for model_name in SUPPORTED_MODELS if model_name not in model_payload]
    if missing_models:
        raise ValueError(f"Faltam parametros para os modelos obrigatorios: {missing_models}")

    parsed_params: dict[str, dict[str, Any]] = {}
    for model_name in SUPPORTED_MODELS:
        model_params = model_payload[model_name]
        if not isinstance(model_params, dict):
            raise ValueError(
                f"Os parametros do modelo '{model_name}' devem ser um objeto JSON."
            )
        parsed_params[model_name] = dict(model_params)

    return parsed_params


def _ensure_results_dirs(results_root: Path) -> None:
    for folder in ("csv", "json", "md", "plots"):
        (results_root / folder).mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def _save_records(records: list[dict[str, Any]], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    frame = pd.DataFrame(records)
    frame.to_csv(csv_path, index=False)
    _save_json(json_path, records)


def available_datasets(data_dir: Path) -> list[str]:
    datasets: list[str] = []
    for file in data_dir.glob("*-h5.npy"):
        dataset = file.name.replace("-h5.npy", "")
        if (data_dir / f"{dataset}-adj_mx.npy").exists():
            datasets.append(dataset)

    for file in data_dir.glob("*-adj_mx.npy"):
        dataset = file.name.replace("-adj_mx.npy", "")
        base_dataset_name = _infer_base_dataset_name(dataset)
        has_dataset_h5 = (data_dir / f"{dataset}-h5.npy").exists()
        has_base_h5 = (
            base_dataset_name is not None
            and (data_dir / f"{base_dataset_name}-h5.npy").exists()
        )
        if has_dataset_h5 or has_base_h5:
            datasets.append(dataset)

    return sorted(set(datasets))


def resolve_dataset_paths(dataset_name: str, data_dir: Path) -> tuple[Path, Path]:
    base_dataset_name = _infer_base_dataset_name(dataset_name)

    data_path = data_dir / f"{dataset_name}-h5.npy"
    adj_path = data_dir / f"{dataset_name}-adj_mx.npy"

    if not data_path.exists() and base_dataset_name is not None:
        fallback_data_path = data_dir / f"{base_dataset_name}-h5.npy"
        if fallback_data_path.exists():
            data_path = fallback_data_path

    if not data_path.exists() or not adj_path.exists():
        available = ", ".join(available_datasets(data_dir)) or "(nenhum encontrado)"
        raise FileNotFoundError(
            f"Arquivos do dataset '{dataset_name}' nao encontrados em {data_dir}. "
            f"Esperado: '{data_path.name}' e '{adj_path.name}'. Disponiveis: {available}"
        )

    return data_path, adj_path


def _instantiate_model(
    *,
    model_name: str,
    params: dict[str, Any],
    adj_mx: np.ndarray,
    num_nodes: int,
    config: SimpleConfig,
):
    common_kwargs = {
        "adj_mx": adj_mx,
        "num_nodes": num_nodes,
        "input_dim": int(params.get("input_dim", 1)),
        "hidden_dim": int(params.get("hidden_dim", 64)),
        "output_dim": int(params.get("output_dim", 1)),
        "seq_len": config.seq_len,
        "horizon": config.horizon,
        "dropout": float(params.get("dropout", 0.1)),
        "lr": float(params.get("lr", 1e-3)),
        "weight_decay": float(params.get("weight_decay", 1e-4)),
        "epochs": int(params.get("epochs", config.epochs)),
        "patience": int(params.get("patience", 5)),
        "device": config.device,
    }

    if model_name == "GraphWaveNet":
        model = GraphWaveNet(
            **common_kwargs,
            num_blocks=int(params.get("num_blocks", 3)),
            dilation_base=int(params.get("dilation_base", 2)),
            k=int(params.get("k", 2)),
        )
    elif model_name == "MTGNN":
        model = MTGNN(
            **common_kwargs,
            num_blocks=int(params.get("num_blocks", 3)),
            kernel_size=int(params.get("kernel_size", 2)),
            dilation_base=int(params.get("dilation_base", 2)),
            gcn_depth=int(params.get("gcn_depth", 2)),
            propalpha=float(params.get("propalpha", 0.05)),
            node_dim=int(params.get("node_dim", 16)),
        )
    elif model_name == "STICformer":
        model = STICformer(
            **common_kwargs,
            num_layers=int(params.get("num_layers", 2)),
            num_heads=int(params.get("num_heads", 4)),
            ff_multiplier=int(params.get("ff_multiplier", 2)),
        )
    else:
        raise ValueError(f"Modelo nao suportado no pipeline simples: {model_name}")

    model.save_best_model = False
    return model


def _collect_predictions(model, loader):
    predictions = model.predict(loader).detach().cpu().float()
    targets = torch.cat([y for _, y in loader], dim=0).detach().cpu().float()
    return predictions, targets


def _evaluate_loader_metrics(
    *,
    model,
    loader,
    normalization_stats: dict[str, Any] | None,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    loss_value = float(model.evaluate(loader))
    predictions, targets = _collect_predictions(model, loader)
    predictions_np, targets_np = denormalize_arrays(
        predictions,
        targets,
        normalization_stats,
    )
    metrics = compute_regression_metrics(targets_np, predictions_np)
    return loss_value, metrics, predictions_np, targets_np


def _build_validation_summary(
    *,
    params: dict[str, Any],
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_dicts = []
    val_losses = []
    for result in seed_results:
        metric_dicts.append(
            {
                metric_name.replace("val_", "", 1): float(metric_value)
                for metric_name, metric_value in result.items()
                if metric_name.startswith("val_") and metric_name != "val_loss_normalized"
            }
        )
        if "val_loss_normalized" in result:
            val_losses.append(float(result["val_loss_normalized"]))

    summary = {
        "params": dict(params),
        "num_completed_seeds": len(seed_results),
        "seeds": [int(result["seed"]) for result in seed_results],
    }
    summary.update(
        summarize_metric_dicts(
            metric_dicts,
            source_metric_names=REGRESSION_METRICS,
            output_prefix="val",
        )
    )
    if val_losses:
        for stat_name, stat_value in confidence_interval_95(val_losses).items():
            summary[f"val_loss_normalized_{stat_name}"] = stat_value

    return summary


def _build_final_summary(
    *,
    experiment_name: str,
    model_name: str,
    dataset_name: str,
    selection_metric: str,
    params: dict[str, Any],
    selected_config: dict[str, Any],
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    test_metric_dicts = []
    test_losses = []
    for result in seed_results:
        test_metric_dicts.append(
            {
                metric_name.replace("test_", "", 1): float(metric_value)
                for metric_name, metric_value in result.items()
                if metric_name.startswith("test_") and metric_name != "test_loss_normalized"
            }
        )
        if "test_loss_normalized" in result:
            test_losses.append(float(result["test_loss_normalized"]))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "selection_metric": selection_metric,
        "selected_params": dict(params),
        "selected_num_completed_seeds": int(selected_config["num_completed_seeds"]),
        "final_num_completed_seeds": len(seed_results),
    }
    summary.update(
        summarize_metric_dicts(
            test_metric_dicts,
            source_metric_names=REGRESSION_METRICS,
            output_prefix="test",
        )
    )
    if test_losses:
        for stat_name, stat_value in confidence_interval_95(test_losses).items():
            summary[f"test_loss_normalized_{stat_name}"] = stat_value

    summary["selected_config_summary"] = selected_config
    return summary


def run_model_experiment(
    *,
    dataset_name: str,
    experiment_type: str,
    model_name: str,
    params: dict[str, Any],
    train_loader,
    val_loader,
    test_loader,
    adj_mx: np.ndarray,
    num_nodes: int,
    normalization_stats: dict[str, Any],
    config: SimpleConfig,
) -> dict[str, Any]:
    experiment_prefix = "simple" if experiment_type == "original" else f"simple_{experiment_type}"
    experiment_name = f"{experiment_prefix}_{dataset_name}_{model_name}_{config.run_label}"
    seed_results: list[dict[str, Any]] = []

    print(f"\n{'#' * 90}")
    print(f"Iniciando experimento simples: {experiment_name}")
    print(f"Modelo: {model_name} | Device: {config.device}")
    print(f"{'#' * 90}")

    for seed in config.seeds:
        _set_fast_single_seed(seed)
        run_name = f"{experiment_name}_final_seed{seed}"

        model = _instantiate_model(
            model_name=model_name,
            params=params,
            adj_mx=adj_mx,
            num_nodes=num_nodes,
            config=config,
        )

        print(f"\nIniciando treinamento de {model_name} | seed={seed}")
        model.fit(train_loader, val_loader)

        result = {
            "run_name": run_name,
            "seed": seed,
            "phase": "final",
            "params": dict(params),
            "train_epochs_completed": len(getattr(model, "train_losses", []) or []),
            "train_loss_last": float((getattr(model, "train_losses", []) or [np.nan])[-1]),
        }

        val_loss, val_metrics, _, _ = _evaluate_loader_metrics(
            model=model,
            loader=val_loader,
            normalization_stats=normalization_stats,
        )
        result["val_loss_normalized"] = val_loss
        result.update(prefix_metrics(val_metrics, "val"))

        test_loss, test_metrics, test_predictions, test_targets = _evaluate_loader_metrics(
            model=model,
            loader=test_loader,
            normalization_stats=normalization_stats,
        )
        result["test_loss_normalized"] = test_loss
        result.update(prefix_metrics(test_metrics, "test"))

        if config.generate_plots:
            plot_dir = config.results_dir / "plots" / run_name
            report = generate_model_diagnostics(
                predictions=test_predictions,
                targets=test_targets,
                output_dir=plot_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                experiment_name=run_name,
                train_losses=getattr(model, "train_losses", []) or [],
                val_losses=getattr(model, "val_losses", []) or [],
                num_nodes_to_plot=config.plots_num_nodes,
                horizon_for_line_and_heatmap=0,
                max_points_line=max(150, config.plots_max_time_points),
                max_time_points_heatmap=config.plots_max_time_points,
                results_root=config.results_dir,
            )
            print(f"📈 Plots salvos em: {report['output_dir']}")

        seed_results.append(result)

        print(
            "✅ Execucao concluida: "
            f"seed={seed}, val_mae={result.get('val_mae', float('nan')):.4f}, "
            f"test_mae={result.get('test_mae', float('nan')):.4f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_config = _build_validation_summary(params=params, seed_results=seed_results)
    final_summary = _build_final_summary(
        experiment_name=experiment_name,
        model_name=model_name,
        dataset_name=dataset_name,
        selection_metric=config.selection_metric,
        params=params,
        selected_config=selected_config,
        seed_results=seed_results,
    )

    csv_dir = config.results_dir / "csv"
    json_dir = config.results_dir / "json"
    final_csv = csv_dir / f"{experiment_name}_final_test_results.csv"
    final_json = json_dir / f"{experiment_name}_final_test_results.json"
    _save_records(seed_results, final_csv, final_json)

    config_summary_csv = csv_dir / f"{experiment_name}_config_summaries.csv"
    config_summary_json = json_dir / f"{experiment_name}_config_summaries.json"
    _save_records([selected_config], config_summary_csv, config_summary_json)

    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "selection_metric": config.selection_metric,
        "seeds": list(config.seeds),
        "params": dict(params),
        "final_test_results_file": str(final_json),
        "config_summaries_file": str(config_summary_json),
        "selected_config": selected_config,
        "final_summary": final_summary,
    }
    summary_file = json_dir / f"{experiment_name}_summary.json"
    _save_json(summary_file, summary_payload)
    print(f"💾 Resumo salvo em: {summary_file}")

    return {
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "trial_results": list(seed_results),
        "config_summaries": [selected_config],
        "selected_config": selected_config,
        "final_test_results": list(seed_results),
        "final_summary": final_summary,
        "metadata": {
            "pipeline": "simple_pipeline",
            "experiment_type": experiment_type,
            "base_dataset": _infer_base_dataset_name(dataset_name) or dataset_name,
            "device": config.device,
            "run_label": config.run_label,
            "selection_metric": config.selection_metric,
            "seq_len": config.seq_len,
            "horizon": config.horizon,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "seeds": list(config.seeds),
            "normalization_stats": normalization_stats,
            "params": dict(params),
        },
    }


def consolidate_outputs(
    *,
    experiments_data: list[dict[str, Any]],
    output_scope: str,
    config: SimpleConfig,
) -> None:
    prefix = f"{output_scope}_{config.run_label}"
    consolidated_df = consolidate_experiment_results(
        experiments_data=experiments_data,
        output_csv=f"{prefix}_consolidated_experiments.csv",
        output_json=f"{prefix}_consolidated_experiments.json",
        primary_metric="test_mae_mean",
        save_path=config.results_dir,
    )

    if consolidated_df.empty:
        print("Consolidacao vazia. Nenhum relatorio adicional foi gerado.")
        return

    create_comparison_report(
        consolidated_df=consolidated_df,
        output_file=f"{prefix}_comparison_report.md",
        save_path=config.results_dir,
    )

    print("\nArquivos consolidados gerados em:")
    print(f"- {config.results_dir / 'csv' / f'{prefix}_consolidated_experiments.csv'}")
    print(f"- {config.results_dir / 'json' / f'{prefix}_consolidated_experiments.json'}")
    print(f"- {config.results_dir / 'md' / f'{prefix}_comparison_report.md'}")


def run_dataset_pipeline(
    *,
    dataset_name: str,
    experiment_type: str,
    config: SimpleConfig,
    params_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    print(f"\n{'=' * 100}")
    print(f"Tipo de experimento: {experiment_type}")
    print(f"Dataset selecionado: {dataset_name}")
    print(f"Diretorio de dados: {config.data_dir}")
    print(f"Diretorio de resultados: {config.results_dir}")
    print(f"Device: {config.device}")
    print(f"{'=' * 100}")

    data_path, adj_path = resolve_dataset_paths(dataset_name, config.data_dir)
    print(f"Arquivo de serie temporal: {data_path}")
    print(f"Arquivo de adjacencia: {adj_path}")
    expected_dataset_data = config.data_dir / f"{dataset_name}-h5.npy"
    if data_path != expected_dataset_data:
        print(
            "Serie temporal especifica do backbone nao encontrada; "
            f"usando serie temporal base: {data_path.name}"
        )

    data = np.load(data_path)
    adj = np.load(adj_path)

    train_loader, val_loader, test_loader, num_nodes, adj_mx, normalization_stats = (
        prepare_dataloaders_from_arrays(
            data=data,
            adj_mx=adj,
            seq_len=config.seq_len,
            horizon=config.horizon,
            batch_size=config.batch_size,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            normalize=config.normalize,
            normalization_method=config.normalization_method,
            num_workers=4,
            pin_memory=config.device == "cuda",
        )
    )

    experiments_data: list[dict[str, Any]] = []
    for model_name in SUPPORTED_MODELS:
        try:
            experiment_result = run_model_experiment(
                dataset_name=dataset_name,
                experiment_type=experiment_type,
                model_name=model_name,
                params=params_by_model[model_name],
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                normalization_stats=normalization_stats,
                config=config,
            )
            experiments_data.append(experiment_result)
        except Exception as exc:
            print(f"\nFalha ao executar {model_name} para dataset '{dataset_name}': {exc}")
            raise

    consolidate_outputs(
        experiments_data=experiments_data,
        output_scope=dataset_name,
        config=config,
    )
    return experiments_data


def _save_run_metadata(
    *,
    config: SimpleConfig,
    params_by_model: dict[str, dict[str, Any]],
) -> None:
    metadata_path = config.results_dir / "json" / f"{config.run_label}_run_metadata.json"
    _save_json(
        metadata_path,
        {
            "timestamp": datetime.now().isoformat(),
            "pipeline": "simple_pipeline",
            "config": asdict(config),
            "params_by_model": params_by_model,
        },
    )
    print(f"🧾 Metadata da execucao salva em: {metadata_path}")


def _global_output_scope(*, experiment_type: str, config: SimpleConfig) -> str:
    if experiment_type == "original" and not config.run_backbone:
        return "all-datasets"
    return f"{experiment_type}_all-datasets"


def run_experiment_group(
    *,
    experiment_type: str,
    dataset_names: list[str],
    config: SimpleConfig,
    params_by_model: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    print(f"\n{'=' * 100}")
    print(f"Iniciando grupo simples: {experiment_type}")
    print(f"Datasets: {dataset_names}")
    print(f"{'=' * 100}")

    all_experiments: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        dataset_results = run_dataset_pipeline(
            dataset_name=dataset_name,
            experiment_type=experiment_type,
            config=config,
            params_by_model=params_by_model,
        )
        all_experiments.extend(dataset_results)

    if len(dataset_names) > 1 and all_experiments:
        consolidate_outputs(
            experiments_data=all_experiments,
            output_scope=_global_output_scope(
                experiment_type=experiment_type,
                config=config,
            ),
            config=config,
        )

    return all_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline simples para 3 modelos de forecasting.")
    parser.add_argument(
        "--config",
        default="simple_config.json",
        help="Caminho para o arquivo JSON de configuracao do pipeline simples.",
    )
    parser.add_argument(
        "--params",
        default="params.json",
        help="Caminho para o arquivo JSON com os hiperparametros dos modelos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_file = Path(args.config).resolve()
    params_file = Path(args.params).resolve()

    config = load_simple_config(config_file, params_file)
    params_by_model = load_model_params(params_file)

    _ensure_results_dirs(config.results_dir)
    _save_run_metadata(config=config, params_by_model=params_by_model)

    print("Configuracao do pipeline simples:")
    print(f"- CONFIG_FILE: {config.config_file}")
    print(f"- PARAMS_FILE: {config.params_file}")
    print(f"- EXPERIMENT: {config.experiment}")
    print(f"- DATASETS: {config.dataset_names}")
    if config.run_backbone:
        print(f"- BACKBONE_DATASETS: {config.backbone_dataset_names}")
        print(f"- BACKBONE_METHODS: {config.backbone_methods}")
        print(f"- BACKBONE_ALPHA: {config.backbone_alpha}")
    print(f"- DEVICE: {config.device}")
    print(f"- RESULTS_DIR: {config.results_dir}")
    print(f"- SEEDS: {config.seeds} (single-seed fast mode)")
    print(f"- RUN_LABEL: {config.run_label}")
    print(f"- MODELS: {list(SUPPORTED_MODELS)}")

    all_experiments: list[dict[str, Any]] = []
    if config.run_original:
        all_experiments.extend(
            run_experiment_group(
                experiment_type="original",
                dataset_names=config.dataset_names,
                config=config,
                params_by_model=params_by_model,
            )
        )

    if config.run_backbone:
        all_experiments.extend(
            run_experiment_group(
                experiment_type="backbone",
                dataset_names=config.backbone_dataset_names,
                config=config,
                params_by_model=params_by_model,
            )
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime
from itertools import product
import json
from pathlib import Path
import traceback

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch

from models.WaveNet import GraphWaveNet
from models.DCRNN import DCRNN
from models.MTGNN import MTGNN
from models.DGCRN import DGCRN
from models.STICformer import STICformer
from models.PatchSTG import PatchSTG
from shared.metrics import (
    REGRESSION_METRICS,
    compute_regression_metrics,
    confidence_interval_95,
    denormalize_arrays,
    prefix_metrics,
    summarize_metric_dicts,
)
from shared.reproducibility import set_global_seed
from shared.visualization import generate_model_diagnostics


RESULTS_DIR: Path = Path("results")
CSV_DIR: Path = RESULTS_DIR / "csv"
JSON_DIR: Path = RESULTS_DIR / "json"
MD_DIR: Path = RESULTS_DIR / "md"
PLOTS_DIR: Path = RESULTS_DIR / "plots"
BEST_MODELS_DIR: Path = RESULTS_DIR / "best-models"


MODEL_ARTIFACT_NAMES = {
    "GraphWaveNet": "graph_wavenet_model",
    "DCRNN": "dcrnn_model",
    "MTGNN": "mtgnn_model",
    "DGCRN": "dgcrn_model",
    "STICformer": "sticformer_model",
    "PatchSTG": "patchstg_model",
}


def set_results_root(results_root: str | Path) -> None:
    global RESULTS_DIR, CSV_DIR, JSON_DIR, MD_DIR, PLOTS_DIR, BEST_MODELS_DIR

    RESULTS_DIR = Path(results_root)
    CSV_DIR = RESULTS_DIR / "csv"
    JSON_DIR = RESULTS_DIR / "json"
    MD_DIR = RESULTS_DIR / "md"
    PLOTS_DIR = RESULTS_DIR / "plots"
    BEST_MODELS_DIR = RESULTS_DIR / "best-models"

    for _dir in (RESULTS_DIR, CSV_DIR, JSON_DIR, MD_DIR, PLOTS_DIR, BEST_MODELS_DIR):
        _dir.mkdir(parents=True, exist_ok=True)


def _stable_params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _configure_best_model_path(model, model_name: str, run_name: str) -> Path:
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "no-run-id"
    model_dir = BEST_MODELS_DIR / run_name / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"best_model_{model_name.lower()}.pth"
    model.best_model_path = str(best_model_path)
    return best_model_path


def _collect_predictions(model, loader):
    preds = model.predict(loader).detach().cpu().float()
    targets = torch.cat([y for _, y in loader], dim=0).detach().cpu().float()
    return preds, targets


def _evaluate_loader_metrics(
    *,
    model,
    loader,
    normalization_stats: dict | None,
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    loss_value = float(model.evaluate(loader))
    preds, targets = _collect_predictions(model, loader)
    preds_np, targets_np = denormalize_arrays(preds, targets, normalization_stats)
    metrics = compute_regression_metrics(targets_np, preds_np)
    return loss_value, metrics, preds, targets, preds_np, targets_np


def _generate_and_log_plots(
    *,
    preds_np: np.ndarray,
    targets_np: np.ndarray,
    model,
    model_name: str,
    dataset_name: str,
    run_name: str,
    generate_plots: bool,
    num_nodes_to_plot: int,
    max_time_points: int,
) -> None:
    if not generate_plots:
        return

    plot_dir = PLOTS_DIR / run_name
    train_losses = getattr(model, "train_losses", []) or []
    val_losses = getattr(model, "val_losses", []) or []

    report = generate_model_diagnostics(
        predictions=preds_np,
        targets=targets_np,
        output_dir=plot_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        experiment_name=run_name,
        train_losses=train_losses,
        val_losses=val_losses,
        num_nodes_to_plot=num_nodes_to_plot,
        horizon_for_line_and_heatmap=0,
        max_points_line=max(150, max_time_points),
        max_time_points_heatmap=max_time_points,
        results_root=RESULTS_DIR,
    )

    mlflow.log_artifacts(str(plot_dir), artifact_path="plots")
    for file_path in report["generated_files"]:
        path = Path(file_path)
        if path.exists() and plot_dir not in path.parents:
            mlflow.log_artifact(str(path), artifact_path=path.parent.name)

    print(f"📈 Plots salvos em: {report['output_dir']}")


def _instantiate_model(
    *,
    model_name: str,
    params: dict,
    adj_mx,
    num_nodes: int,
    device: str,
):
    common_kwargs = {
        "adj_mx": adj_mx,
        "num_nodes": num_nodes,
        "input_dim": params["input_dim"],
        "hidden_dim": params["hidden_dim"],
        "output_dim": params["output_dim"],
        "seq_len": params["seq_len"],
        "horizon": params["horizon"],
        "dropout": params.get("dropout", 0.1),
        "lr": params.get("lr", 1e-3),
        "weight_decay": params.get("weight_decay", 1e-4),
        "epochs": params.get("epochs", 50),
        "patience": params.get("patience", 10),
        "device": device,
    }

    if model_name == "GraphWaveNet":
        return GraphWaveNet(
            **common_kwargs,
            num_blocks=params.get("num_blocks", 4),
            dilation_base=params.get("dilation_base", 2),
            k=params.get("k", 2),
        )

    if model_name == "DCRNN":
        return DCRNN(
            **common_kwargs,
            k=params.get("k", 2),
            use_scheduled_sampling=params.get("use_scheduled_sampling", False),
            teacher_forcing_ratio=params.get("teacher_forcing_ratio", 0.5),
        )

    if model_name == "MTGNN":
        return MTGNN(
            **common_kwargs,
            num_blocks=params.get("num_blocks", 3),
            kernel_size=params.get("kernel_size", 2),
            dilation_base=params.get("dilation_base", 2),
            gcn_depth=params.get("gcn_depth", 2),
            propalpha=params.get("propalpha", 0.05),
            node_dim=params.get("node_dim", 16),
        )

    if model_name == "DGCRN":
        return DGCRN(
            **common_kwargs,
            node_dim=params.get("node_dim", 16),
            gcn_depth=params.get("gcn_depth", 2),
        )

    if model_name == "STICformer":
        return STICformer(
            **common_kwargs,
            num_layers=params.get("num_layers", 2),
            num_heads=params.get("num_heads", 4),
            ff_multiplier=params.get("ff_multiplier", 2),
        )

    if model_name == "PatchSTG":
        return PatchSTG(
            **common_kwargs,
            patch_len=params.get("patch_len", 4),
            patch_stride=params.get("patch_stride", 2),
            num_layers=params.get("num_layers", 2),
            num_heads=params.get("num_heads", 4),
            ff_multiplier=params.get("ff_multiplier", 2),
        )

    raise ValueError(f"Modelo nao suportado: {model_name}")


def _run_single_training(
    *,
    model_name: str,
    params: dict,
    train_loader,
    val_loader,
    test_loader=None,
    adj_mx=None,
    num_nodes: int,
    experiment_name: str,
    dataset_name: str,
    seed: int,
    phase: str,
    device: str,
    normalization_stats: dict | None,
    generate_plots: bool,
    num_nodes_to_plot: int,
    max_time_points: int,
) -> dict:
    set_global_seed(seed)
    run_name = f"{experiment_name}_{phase}_seed{seed}"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "phase": phase,
                "seed": seed,
                "evaluation_scale": "original",
            }
        )
        mlflow.log_params(params)
        mlflow.log_param("device", device)
        mlflow.log_param("num_nodes", num_nodes)
        mlflow.log_param("seed", seed)

        model = _instantiate_model(
            model_name=model_name,
            params=params,
            adj_mx=adj_mx,
            num_nodes=num_nodes,
            device=device,
        )
        best_model_path = _configure_best_model_path(model, model_name, run_name)
        mlflow.log_param("best_model_path", str(best_model_path))

        print(f"\nIniciando treinamento de {model_name} | fase={phase} | seed={seed}")
        model.fit(train_loader, val_loader)

        result = {
            "run_name": run_name,
            "seed": seed,
            "phase": phase,
            "params": dict(params),
            "mlflow_run_id": mlflow.active_run().info.run_id if mlflow.active_run() else None,
            "train_epochs_completed": len(getattr(model, "train_losses", []) or []),
            "train_loss_last": float((getattr(model, "train_losses", []) or [np.nan])[-1]),
        }

        if val_loader is not None:
            (
                val_loss,
                val_metrics,
                _,
                _,
                _,
                _,
            ) = _evaluate_loader_metrics(
                model=model,
                loader=val_loader,
                normalization_stats=normalization_stats,
            )
            result["val_loss_normalized"] = val_loss
            result.update(prefix_metrics(val_metrics, "val"))

            mlflow.log_metric("val_loss_normalized", val_loss)
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", metric_value)

        if test_loader is not None:
            (
                test_loss,
                test_metrics,
                _,
                _,
                test_preds_np,
                test_targets_np,
            ) = _evaluate_loader_metrics(
                model=model,
                loader=test_loader,
                normalization_stats=normalization_stats,
            )
            result["test_loss_normalized"] = test_loss
            result.update(prefix_metrics(test_metrics, "test"))

            mlflow.log_metric("test_loss_normalized", test_loss)
            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            _generate_and_log_plots(
                preds_np=test_preds_np,
                targets_np=test_targets_np,
                model=model,
                model_name=model_name,
                dataset_name=dataset_name,
                run_name=run_name,
                generate_plots=generate_plots,
                num_nodes_to_plot=num_nodes_to_plot,
                max_time_points=max_time_points,
            )

            mlflow.pytorch.log_model(model, MODEL_ARTIFACT_NAMES[model_name])

        return result


def _group_trials_by_params(trial_results: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}

    for trial in trial_results:
        key = _stable_params_key(trial["params"])
        bucket = grouped.setdefault(
            key,
            {
                "params": trial["params"],
                "trials": [],
            },
        )
        bucket["trials"].append(trial)

    summaries: list[dict] = []
    for bucket in grouped.values():
        trials = bucket["trials"]
        metric_dicts = []
        for trial in trials:
            metrics = {
                metric_name: float(trial[metric_name])
                for metric_name in trial
                if metric_name.startswith("val_") and metric_name != "val_loss_normalized"
            }
            cleaned = {key.replace("val_", "", 1): value for key, value in metrics.items()}
            metric_dicts.append(cleaned)

        summary = {
            "params": bucket["params"],
            "num_completed_seeds": len(trials),
            "seeds": [int(trial["seed"]) for trial in trials],
        }
        summary.update(
            summarize_metric_dicts(
                metric_dicts,
                source_metric_names=REGRESSION_METRICS,
                output_prefix="val",
            )
        )

        val_losses = [float(trial["val_loss_normalized"]) for trial in trials if "val_loss_normalized" in trial]
        if val_losses:
            for stat_name, stat_value in confidence_interval_95(val_losses).items():
                summary[f"val_loss_normalized_{stat_name}"] = stat_value

        summaries.append(summary)

    return summaries


def _select_best_config(
    *,
    config_summaries: list[dict],
    selection_metric: str,
) -> dict | None:
    if not config_summaries:
        return None

    ordered = sorted(
        config_summaries,
        key=lambda item: (
            item["num_completed_seeds"] == 0,
            float(item.get(f"{selection_metric}_mean", float("inf"))),
            float(item.get("val_rmse_mean", float("inf"))),
        ),
    )
    return ordered[0]


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _save_records(records: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(records)
    frame.to_csv(csv_path, index=False)
    _save_json(json_path, records)


def _save_search_artifacts(
    *,
    experiment_name: str,
    model_name: str,
    dataset_name: str,
    selection_metric: str,
    seeds: list[int],
    trial_results: list[dict],
    config_summaries: list[dict],
    selected_config: dict | None,
) -> dict:
    search_csv = CSV_DIR / f"{experiment_name}_trial_results.csv"
    search_json = JSON_DIR / f"{experiment_name}_trial_results.json"
    _save_records(trial_results, search_csv, search_json)

    config_csv = CSV_DIR / f"{experiment_name}_config_summaries.csv"
    config_json = JSON_DIR / f"{experiment_name}_config_summaries.json"
    _save_records(config_summaries, config_csv, config_json)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "selection_metric": selection_metric,
        "seeds": seeds,
        "trial_results_file": str(search_json),
        "config_summaries_file": str(config_json),
        "selected_config": selected_config,
    }
    summary_file = JSON_DIR / f"{experiment_name}_search_summary.json"
    _save_json(summary_file, payload)
    payload["summary_file"] = str(summary_file)
    print(f"\n💾 Resumo do search salvo em: {summary_file}")
    return payload


def _build_final_summary(
    *,
    experiment_name: str,
    model_name: str,
    dataset_name: str,
    selection_metric: str,
    selected_config: dict,
    final_test_results: list[dict],
) -> dict:
    test_metric_dicts = []
    for result in final_test_results:
        metrics = {
            metric_name.replace("test_", "", 1): float(result[metric_name])
            for metric_name in result
            if metric_name.startswith("test_") and metric_name != "test_loss_normalized"
        }
        test_metric_dicts.append(metrics)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "selection_metric": selection_metric,
        "selected_params": dict(selected_config["params"]),
        "selected_num_completed_seeds": int(selected_config["num_completed_seeds"]),
        "final_num_completed_seeds": len(final_test_results),
    }

    summary.update(
        summarize_metric_dicts(
            test_metric_dicts,
            source_metric_names=REGRESSION_METRICS,
            output_prefix="test",
        )
    )

    test_losses = [
        float(result["test_loss_normalized"])
        for result in final_test_results
        if "test_loss_normalized" in result
    ]
    if test_losses:
        for stat_name, stat_value in confidence_interval_95(test_losses).items():
            summary[f"test_loss_normalized_{stat_name}"] = stat_value

    summary["selected_config_summary"] = selected_config
    return summary


def _run_grid_search(
    *,
    model_name: str,
    param_grid: dict,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes: int,
    experiment_name: str,
    dataset_name: str,
    device: str,
    normalization_stats: dict | None,
    seeds: list[int],
    selection_metric: str,
    generate_plots: bool,
    num_nodes_to_plot: int,
    max_time_points: int,
    run_final_stage: bool = True,
) -> dict:
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    search_trial_results: list[dict] = []

    for combination in product(*values):
        params = dict(zip(keys, combination))
        print(f"\n{'=' * 60}")
        print(f"Testando combinação ({model_name}): {params}")
        print(f"{'=' * 60}")

        for seed in seeds:
            try:
                trial_result = _run_single_training(
                    model_name=model_name,
                    params=params,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=None,
                    adj_mx=adj_mx,
                    num_nodes=num_nodes,
                    experiment_name=experiment_name,
                    dataset_name=dataset_name,
                    seed=seed,
                    phase="search",
                    device=device,
                    normalization_stats=normalization_stats,
                    generate_plots=False,
                    num_nodes_to_plot=num_nodes_to_plot,
                    max_time_points=max_time_points,
                )
                search_trial_results.append(trial_result)
                print(
                    "✅ Search concluído: "
                    f"seed={seed}, val_mae={trial_result.get('val_mae', float('nan')):.4f}, "
                    f"val_rmse={trial_result.get('val_rmse', float('nan')):.4f}"
                )
            except Exception as exc:
                print(f"Tipo do erro: {type(exc).__name__}")
                print(f"Mensagem: {str(exc)}")
                print(f"\n{'=' * 60}")
                print("TRACEBACK COMPLETO:")
                print(f"{'=' * 60}")
                traceback.print_exc()

    if not search_trial_results:
        print("❌ Nenhum resultado obtido no search!")
        return {
            "trial_results": [],
            "config_summaries": [],
            "selected_config": None,
            "final_test_results": [],
            "final_summary": None,
        }

    config_summaries = _group_trials_by_params(search_trial_results)
    selected_config = _select_best_config(
        config_summaries=config_summaries,
        selection_metric=selection_metric,
    )

    if selected_config is None:
        print("❌ Nenhuma configuração selecionada!")
        return {
            "trial_results": search_trial_results,
            "config_summaries": config_summaries,
            "selected_config": None,
            "final_test_results": [],
            "final_summary": None,
        }

    search_summary = _save_search_artifacts(
        experiment_name=experiment_name,
        model_name=model_name,
        dataset_name=dataset_name,
        selection_metric=selection_metric,
        seeds=seeds,
        trial_results=search_trial_results,
        config_summaries=config_summaries,
        selected_config=selected_config,
    )

    selected_params = selected_config["params"]
    print(f"\n{'=' * 80}")
    print(f"Melhor configuração por {selection_metric}: {selected_params}")
    print(f"Valor médio: {selected_config.get(f'{selection_metric}_mean', float('nan')):.4f}")
    print(f"{'=' * 80}")

    if not run_final_stage:
        return {
            "trial_results": search_trial_results,
            "config_summaries": config_summaries,
            "selected_config": selected_config,
            "final_test_results": [],
            "final_summary": None,
            "search_summary": search_summary,
        }

    final_test_results: list[dict] = []
    for seed in seeds:
        try:
            final_result = _run_single_training(
                model_name=model_name,
                params=selected_params,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                seed=seed,
                phase="final",
                device=device,
                normalization_stats=normalization_stats,
                generate_plots=generate_plots,
                num_nodes_to_plot=num_nodes_to_plot,
                max_time_points=max_time_points,
            )
            final_test_results.append(final_result)
            print(
                "✅ Final concluído: "
                f"seed={seed}, test_mae={final_result.get('test_mae', float('nan')):.4f}, "
                f"test_rmse={final_result.get('test_rmse', float('nan')):.4f}"
            )
        except Exception as exc:
            print(f"Tipo do erro: {type(exc).__name__}")
            print(f"Mensagem: {str(exc)}")
            print(f"\n{'=' * 60}")
            print("TRACEBACK COMPLETO:")
            print(f"{'=' * 60}")
            traceback.print_exc()

    if not final_test_results:
        print("❌ Nenhum resultado final de teste obtido!")
        return {
            "trial_results": search_trial_results,
            "config_summaries": config_summaries,
            "selected_config": selected_config,
            "final_test_results": [],
            "final_summary": None,
        }

    final_summary = _build_final_summary(
        experiment_name=experiment_name,
        model_name=model_name,
        dataset_name=dataset_name,
        selection_metric=selection_metric,
        selected_config=selected_config,
        final_test_results=final_test_results,
    )

    final_csv = CSV_DIR / f"{experiment_name}_final_test_results.csv"
    final_json = JSON_DIR / f"{experiment_name}_final_test_results.json"
    _save_records(final_test_results, final_csv, final_json)

    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "selection_metric": selection_metric,
        "seeds": seeds,
        "search_trial_results_file": search_summary["trial_results_file"],
        "config_summaries_file": search_summary["config_summaries_file"],
        "final_test_results_file": str(final_json),
        "config_summaries": config_summaries,
        "selected_config": selected_config,
        "final_summary": final_summary,
    }
    summary_file = JSON_DIR / f"{experiment_name}_summary.json"
    _save_json(summary_file, summary_payload)
    print(f"\n💾 Resumo científico salvo em: {summary_file}")

    return {
        "trial_results": search_trial_results,
        "config_summaries": config_summaries,
        "selected_config": selected_config,
        "final_test_results": final_test_results,
        "final_summary": final_summary,
        "search_summary": search_summary,
    }


def run_selected_model(
    *,
    model_name: str,
    params: dict,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes: int,
    experiment_name: str,
    dataset_name: str,
    device: str,
    normalization_stats: dict | None,
    seeds: list[int],
    selection_metric: str,
    generate_plots: bool,
    num_nodes_to_plot: int,
    max_time_points: int,
    selected_config: dict | None = None,
) -> dict:
    resolved_selected_config = dict(selected_config or {})
    resolved_selected_config["params"] = dict(params)
    resolved_selected_config.setdefault("num_completed_seeds", len(seeds))
    resolved_selected_config.setdefault("seeds", list(seeds))

    final_test_results: list[dict] = []
    for seed in seeds:
        try:
            final_result = _run_single_training(
                model_name=model_name,
                params=params,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_mx=adj_mx,
                num_nodes=num_nodes,
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                seed=seed,
                phase="final",
                device=device,
                normalization_stats=normalization_stats,
                generate_plots=generate_plots,
                num_nodes_to_plot=num_nodes_to_plot,
                max_time_points=max_time_points,
            )
            final_test_results.append(final_result)
            print(
                "✅ Final concluído: "
                f"seed={seed}, test_mae={final_result.get('test_mae', float('nan')):.4f}, "
                f"test_rmse={final_result.get('test_rmse', float('nan')):.4f}"
            )
        except Exception as exc:
            print(f"Tipo do erro: {type(exc).__name__}")
            print(f"Mensagem: {str(exc)}")
            print(f"\n{'=' * 60}")
            print("TRACEBACK COMPLETO:")
            print(f"{'=' * 60}")
            traceback.print_exc()

    if not final_test_results:
        print("❌ Nenhum resultado final de teste obtido!")
        return {
            "trial_results": [],
            "config_summaries": [],
            "selected_config": resolved_selected_config,
            "final_test_results": [],
            "final_summary": None,
        }

    final_summary = _build_final_summary(
        experiment_name=experiment_name,
        model_name=model_name,
        dataset_name=dataset_name,
        selection_metric=selection_metric,
        selected_config=resolved_selected_config,
        final_test_results=final_test_results,
    )

    final_csv = CSV_DIR / f"{experiment_name}_final_test_results.csv"
    final_json = JSON_DIR / f"{experiment_name}_final_test_results.json"
    _save_records(final_test_results, final_csv, final_json)

    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "selection_metric": selection_metric,
        "seeds": seeds,
        "selected_config": resolved_selected_config,
        "final_test_results_file": str(final_json),
        "final_summary": final_summary,
    }
    summary_file = JSON_DIR / f"{experiment_name}_selected_summary.json"
    _save_json(summary_file, summary_payload)
    print(f"\n💾 Resumo da execução selecionada salvo em: {summary_file}")

    return {
        "trial_results": [],
        "config_summaries": [],
        "selected_config": resolved_selected_config,
        "final_test_results": final_test_results,
        "final_summary": final_summary,
    }


def GraphWaveNet_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="GraphWaveNet_GridSearch",
    dataset_name="unknown",
    device="cpu",
    normalization_stats=None,
    seeds=None,
    selection_metric="val_mae",
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
    run_final_stage=True,
):
    return _run_grid_search(
        model_name="GraphWaveNet",
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj_mx=adj_mx,
        num_nodes=num_nodes,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        device=device,
        normalization_stats=normalization_stats,
        seeds=seeds or [42],
        selection_metric=selection_metric,
        generate_plots=generate_plots,
        num_nodes_to_plot=num_nodes_to_plot,
        max_time_points=max_time_points,
        run_final_stage=run_final_stage,
    )


def DCRNN_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="DCRNN_GridSearch",
    dataset_name="unknown",
    device="cpu",
    normalization_stats=None,
    seeds=None,
    selection_metric="val_mae",
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
    run_final_stage=True,
):
    return _run_grid_search(
        model_name="DCRNN",
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj_mx=adj_mx,
        num_nodes=num_nodes,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        device=device,
        normalization_stats=normalization_stats,
        seeds=seeds or [42],
        selection_metric=selection_metric,
        generate_plots=generate_plots,
        num_nodes_to_plot=num_nodes_to_plot,
        max_time_points=max_time_points,
        run_final_stage=run_final_stage,
    )


def MTGNN_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="MTGNN_GridSearch",
    dataset_name="unknown",
    device="cpu",
    normalization_stats=None,
    seeds=None,
    selection_metric="val_mae",
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
    run_final_stage=True,
):
    return _run_grid_search(
        model_name="MTGNN",
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj_mx=adj_mx,
        num_nodes=num_nodes,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        device=device,
        normalization_stats=normalization_stats,
        seeds=seeds or [42],
        selection_metric=selection_metric,
        generate_plots=generate_plots,
        num_nodes_to_plot=num_nodes_to_plot,
        max_time_points=max_time_points,
        run_final_stage=run_final_stage,
    )


def DGCRN_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="DGCRN_GridSearch",
    dataset_name="unknown",
    device="cpu",
    normalization_stats=None,
    seeds=None,
    selection_metric="val_mae",
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
    run_final_stage=True,
):
    return _run_grid_search(
        model_name="DGCRN",
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj_mx=adj_mx,
        num_nodes=num_nodes,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        device=device,
        normalization_stats=normalization_stats,
        seeds=seeds or [42],
        selection_metric=selection_metric,
        generate_plots=generate_plots,
        num_nodes_to_plot=num_nodes_to_plot,
        max_time_points=max_time_points,
        run_final_stage=run_final_stage,
    )


def STICformer_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="STICformer_GridSearch",
    dataset_name="unknown",
    device="cpu",
    normalization_stats=None,
    seeds=None,
    selection_metric="val_mae",
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
    run_final_stage=True,
):
    return _run_grid_search(
        model_name="STICformer",
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj_mx=adj_mx,
        num_nodes=num_nodes,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        device=device,
        normalization_stats=normalization_stats,
        seeds=seeds or [42],
        selection_metric=selection_metric,
        generate_plots=generate_plots,
        num_nodes_to_plot=num_nodes_to_plot,
        max_time_points=max_time_points,
        run_final_stage=run_final_stage,
    )


def PatchSTG_grid_search(
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes,
    experiment_name="PatchSTG_GridSearch",
    dataset_name="unknown",
    device="cpu",
    normalization_stats=None,
    seeds=None,
    selection_metric="val_mae",
    generate_plots=True,
    num_nodes_to_plot=4,
    max_time_points=350,
    run_final_stage=True,
):
    return _run_grid_search(
        model_name="PatchSTG",
        param_grid=param_grid,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adj_mx=adj_mx,
        num_nodes=num_nodes,
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        device=device,
        normalization_stats=normalization_stats,
        seeds=seeds or [42],
        selection_metric=selection_metric,
        generate_plots=generate_plots,
        num_nodes_to_plot=num_nodes_to_plot,
        max_time_points=max_time_points,
        run_final_stage=run_final_stage,
    )

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import traceback

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - fallback para dry-run sem torch
    torch = None

from pipeline.bootstrap import ensure_workspace_root_on_path
from pipeline.config import PipelineConfig
from pipeline.datasets import (
    DatasetGroup,
    infer_base_dataset_name,
    resolve_backbone_dataset_names,
    resolve_dataset_paths,
)
from pipeline.model_registry import build_model_params, build_param_grids, load_grid_search_registry


_SHARED_RUNTIME: dict[str, object] | None = None


def _shared_runtime() -> dict[str, object]:
    global _SHARED_RUNTIME
    if _SHARED_RUNTIME is not None:
        return _SHARED_RUNTIME

    ensure_workspace_root_on_path()
    from shared.MLFlow import run_selected_model, set_results_root  # noqa: E402
    from shared.loaders import prepare_dataloaders_from_arrays  # noqa: E402
    from shared.resultSumarization import (  # noqa: E402
        consolidate_experiment_results,
        consolidate_search_experiment_results,
        create_comparison_report,
        create_search_report,
        export_best_configs_to_json,
    )

    _SHARED_RUNTIME = {
        "run_selected_model": run_selected_model,
        "set_results_root": set_results_root,
        "prepare_dataloaders_from_arrays": prepare_dataloaders_from_arrays,
        "consolidate_experiment_results": consolidate_experiment_results,
        "consolidate_search_experiment_results": consolidate_search_experiment_results,
        "create_comparison_report": create_comparison_report,
        "create_search_report": create_search_report,
        "export_best_configs_to_json": export_best_configs_to_json,
    }
    return _SHARED_RUNTIME


def _load_selected_configs(best_configs_file: Path) -> dict[str, dict[str, dict]]:
    if not best_configs_file.exists():
        raise FileNotFoundError(f"Arquivo de melhores configuracoes nao encontrado: {best_configs_file}")

    with best_configs_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            f"O arquivo {best_configs_file} deve conter um objeto JSON com dataset -> model -> config."
        )
    return payload


def _resolve_selected_model_config(
    *,
    selected_configs_by_dataset: dict[str, dict[str, dict]],
    dataset_name: str,
    model_name: str,
) -> tuple[dict, dict]:
    dataset_candidates = [dataset_name]
    base_dataset_name = infer_base_dataset_name(dataset_name)
    if base_dataset_name and base_dataset_name not in dataset_candidates:
        dataset_candidates.append(base_dataset_name)

    dataset_configs = None
    matched_dataset_name = None
    for candidate in dataset_candidates:
        current = selected_configs_by_dataset.get(candidate)
        if isinstance(current, dict):
            dataset_configs = current
            matched_dataset_name = candidate
            break

    if dataset_configs is None:
        raise KeyError(
            f"Dataset '{dataset_name}' nao encontrado em best_configs. "
            f"Candidatos tentados: {dataset_candidates}"
        )

    selected_config = dataset_configs.get(model_name)
    if not isinstance(selected_config, dict):
        raise KeyError(
            f"Modelo '{model_name}' nao encontrado em best_configs para dataset '{matched_dataset_name}'."
        )

    selected_params = selected_config.get("selected_params")
    if not isinstance(selected_params, dict) or not selected_params:
        raise ValueError(
            f"selected_params invalido para dataset '{matched_dataset_name}' e modelo '{model_name}'."
        )

    resolved_config = dict(selected_config)
    resolved_config["config_source_dataset"] = matched_dataset_name
    return resolved_config, selected_params


def _build_groups(config: PipelineConfig) -> list[DatasetGroup]:
    groups: list[DatasetGroup] = []

    if config.experiment_scope in {"original", "both"}:
        groups.append(
            DatasetGroup(
                experiment_type="original",
                npy_dir=config.original_data_dir,
                dataset_names=config.dataset_names,
            )
        )

    if config.experiment_scope in {"backbone", "both"}:
        backbone_dataset_names = resolve_backbone_dataset_names(
            dataset_names=config.dataset_names,
            npy_dir=config.backbone_data_dir,
            methods=config.backbone_methods,
            alpha=config.backbone_alpha,
            explicit_names=config.backbone_dataset_names,
        )
        groups.append(
            DatasetGroup(
                experiment_type="backbone",
                npy_dir=config.backbone_data_dir,
                dataset_names=backbone_dataset_names,
            )
        )

    return groups


def _prepare_dataloaders(config: PipelineConfig, *, data_path: Path, adj_path: Path):
    import numpy as np

    runtime = _shared_runtime()
    data = np.load(data_path)
    adj = np.load(adj_path)

    return runtime["prepare_dataloaders_from_arrays"](
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
        pin_memory=config.device == "cuda",
    )


def _consolidate_outputs(
    *,
    experiments_data: list[dict],
    output_scope: str,
    results_root: Path,
    run_label: str,
    mode: str,
) -> None:
    runtime = _shared_runtime()
    prefix = f"{output_scope}_{run_label}"

    if mode == "search_best":
        consolidated_df = runtime["consolidate_search_experiment_results"](
            experiments_data=experiments_data,
            output_csv=f"{prefix}_selected_configs.csv",
            output_json=f"{prefix}_selected_configs.json",
            primary_metric="val_mae_mean",
            save_path=results_root,
        )
        if consolidated_df.empty:
            return

        runtime["create_search_report"](
            consolidated_df=consolidated_df,
            output_file=f"{prefix}_selection_report.md",
            save_path=results_root,
        )
        runtime["export_best_configs_to_json"](
            consolidated_df=consolidated_df,
            output_file=f"{prefix}_best_configs.json",
            save_path=results_root,
        )
        return

    consolidated_df = runtime["consolidate_experiment_results"](
        experiments_data=experiments_data,
        output_csv=f"{prefix}_consolidated_experiments.csv",
        output_json=f"{prefix}_consolidated_experiments.json",
        primary_metric="test_mae_mean",
        save_path=results_root,
    )
    if consolidated_df.empty:
        return

    runtime["create_comparison_report"](
        consolidated_df=consolidated_df,
        output_file=f"{prefix}_comparison_report.md",
        save_path=results_root,
    )
    runtime["export_best_configs_to_json"](
        consolidated_df=consolidated_df,
        output_file=f"{prefix}_best_configs.json",
        save_path=results_root,
    )


def _run_single_model(
    *,
    config: PipelineConfig,
    experiment_type: str,
    dataset_name: str,
    model_name: str,
    param_grid: dict,
    configured_params: dict,
    train_loader,
    val_loader,
    test_loader,
    adj_mx,
    num_nodes: int,
    normalization_stats: dict,
    selected_configs_by_dataset: dict[str, dict[str, dict]] | None,
) -> dict | None:
    runtime = _shared_runtime()
    experiment_name = f"{experiment_type}_{dataset_name}_{model_name}_{config.run_label}"

    if config.mode == "run_best":
        if selected_configs_by_dataset is None:
            raise ValueError("selected_configs_by_dataset e obrigatorio no modo run_best.")

        selected_config, selected_params = _resolve_selected_model_config(
            selected_configs_by_dataset=selected_configs_by_dataset,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        selection_metric = str(selected_config.get("selection_metric", config.selection_metric)).strip().lower()

        result = runtime["run_selected_model"](
            model_name=model_name,
            params=selected_params,
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
            selection_metric=selection_metric,
            generate_plots=config.generate_plots,
            num_nodes_to_plot=config.plots_num_nodes,
            max_time_points=config.plots_max_time_points,
            selected_config=selected_config,
        )
    elif config.mode == "run_configured":
        selected_config = {
            "selection_metric": config.selection_metric,
            "config_source": "model_params",
            "num_completed_seeds": len(config.seeds),
            "seeds": list(config.seeds),
        }
        result = runtime["run_selected_model"](
            model_name=model_name,
            params=configured_params,
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
            selected_config=selected_config,
        )
    else:
        grid_search_fn = load_grid_search_registry()[model_name]
        result = grid_search_fn(
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
            run_final_stage=config.mode == "search_and_run",
        )

    if config.mode == "search_best":
        if not result.get("selected_config"):
            return None
    elif not result.get("final_summary"):
        return None

    return {
        "experiment_name": experiment_name,
        "model": model_name,
        "dataset": dataset_name,
        "trial_results": result.get("trial_results", []),
        "config_summaries": result.get("config_summaries", []),
        "selected_config": result.get("selected_config"),
        "final_test_results": result.get("final_test_results", []),
        "final_summary": result.get("final_summary"),
        "search_summary": result.get("search_summary"),
        "metadata": {
            "mode": config.mode,
            "experiment_type": experiment_type,
            "device": config.device,
            "seeds": config.seeds,
            "selection_metric": config.selection_metric,
        },
    }


def _run_group(
    *,
    config: PipelineConfig,
    group: DatasetGroup,
    selected_configs_by_dataset: dict[str, dict[str, dict]] | None,
    dry_run: bool,
) -> list[dict]:
    results_root = config.results_dir / group.experiment_type

    if dry_run:
        print(
            json.dumps(
                {
                    "experiment_type": group.experiment_type,
                    "npy_dir": str(group.npy_dir),
                    "dataset_names": group.dataset_names,
                    "model_names": config.model_names,
                    "results_root": str(results_root),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return []

    _shared_runtime()["set_results_root"](results_root)
    param_grids = build_param_grids(
        seq_len=config.seq_len,
        horizon=config.horizon,
        epochs=config.epochs,
        overrides=config.param_grids,
    )
    model_params = build_model_params(
        seq_len=config.seq_len,
        horizon=config.horizon,
        epochs=config.epochs,
        overrides=config.model_params,
    )

    all_experiments_data: list[dict] = []
    for dataset_name in group.dataset_names:
        data_path, adj_path = resolve_dataset_paths(dataset_name=dataset_name, npy_dir=group.npy_dir)
        (
            train_loader,
            val_loader,
            test_loader,
            num_nodes,
            adj_mx,
            normalization_stats,
        ) = _prepare_dataloaders(config, data_path=data_path, adj_path=adj_path)

        dataset_experiments: list[dict] = []
        for model_name in config.model_names:
            try:
                result = _run_single_model(
                    config=config,
                    experiment_type=group.experiment_type,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    param_grid=param_grids[model_name],
                    configured_params=model_params[model_name],
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    adj_mx=adj_mx,
                    num_nodes=num_nodes,
                    normalization_stats=normalization_stats,
                    selected_configs_by_dataset=selected_configs_by_dataset,
                )
                if result is not None:
                    dataset_experiments.append(result)
            except Exception:
                print(f"Falha em {group.experiment_type}/{dataset_name}/{model_name}")
                traceback.print_exc()

        if dataset_experiments:
            all_experiments_data.extend(dataset_experiments)
            _consolidate_outputs(
                experiments_data=dataset_experiments,
                output_scope=dataset_name,
                results_root=results_root,
                run_label=config.run_label,
                mode=config.mode,
            )

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(group.dataset_names) > 1 and all_experiments_data:
        _consolidate_outputs(
            experiments_data=all_experiments_data,
            output_scope="all-datasets",
            results_root=results_root,
            run_label=config.run_label,
            mode=config.mode,
        )

    return all_experiments_data


def run_pipeline(config: PipelineConfig, *, dry_run: bool = False) -> None:
    selected_configs_by_dataset = (
        _load_selected_configs(config.best_configs_file) if config.best_configs_file else None
    )
    groups = _build_groups(config)

    if dry_run:
        print(json.dumps(asdict(config), indent=2, ensure_ascii=False, default=str))

    for group in groups:
        _run_group(
            config=config,
            group=group,
            selected_configs_by_dataset=selected_configs_by_dataset,
            dry_run=dry_run,
        )

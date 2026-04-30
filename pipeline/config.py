from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - fallback para ambientes sem torch
    torch = None

from pipeline.bootstrap import PRIMARY_DATA_ROOT, WORKSPACE_ROOT
from pipeline.datasets import SUPPORTED_BACKBONE_METHODS, SUPPORTED_BASE_DATASETS
from pipeline.model_registry import SUPPORTED_MODELS


SUPPORTED_MODES = ("search_best", "run_best", "search_and_run", "run_configured")
SUPPORTED_SCOPES = ("original", "backbone", "both")


@dataclass(frozen=True)
class PipelineConfig:
    config_file: Path
    mode: str
    experiment_scope: str
    dataset_names: list[str]
    backbone_dataset_names: list[str]
    backbone_methods: list[str]
    backbone_alpha: float
    original_data_dir: Path
    backbone_data_dir: Path
    results_dir: Path
    best_configs_file: Path | None
    model_names: list[str]
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
    selection_metric: str
    generate_plots: bool
    plots_num_nodes: int
    plots_max_time_points: int
    run_label: str
    param_grids: dict[str, dict[str, list[Any]]]
    model_params: dict[str, dict[str, Any]]


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"O arquivo {path} deve conter um objeto JSON na raiz.")

    return payload


def _resolve_path(raw_value: Any, *, base_dir: Path, default: Path) -> Path:
    if raw_value is None:
        return default.resolve()

    path = Path(str(raw_value).strip())
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_bool(raw_value: Any, *, default: bool) -> bool:
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"Valor booleano invalido: {raw_value!r}")


def _parse_names(raw_value: Any, *, default: list[str]) -> list[str]:
    if raw_value is None:
        return list(default)

    if isinstance(raw_value, str):
        names = [item.strip() for item in raw_value.split(",") if item.strip()]
        return names or list(default)

    if isinstance(raw_value, list):
        names = [str(item).strip() for item in raw_value if str(item).strip()]
        return names or list(default)

    raise ValueError(f"Lista invalida: {raw_value!r}")


def _parse_seeds(raw_value: Any) -> list[int]:
    if raw_value is None:
        return [42]
    if isinstance(raw_value, int):
        return [int(raw_value)]
    if isinstance(raw_value, str):
        return [int(item.strip()) for item in raw_value.split(",") if item.strip()] or [42]
    if isinstance(raw_value, list):
        return [int(item) for item in raw_value] or [42]
    raise ValueError(f"Lista de seeds invalida: {raw_value!r}")


def _resolve_device(raw_value: Any) -> str:
    requested = str(raw_value or "").strip().lower()
    if not requested:
        requested = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
        return "cpu"
    return requested


def _validate_ratio_sum(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio deve ser igual a 1.0. "
            f"Recebido: {total}"
        )


def load_config(config_file: str | Path) -> PipelineConfig:
    path = Path(config_file).resolve()
    payload = _load_json_object(path)
    base_dir = path.parent

    mode = str(payload.get("mode", "search_best")).strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"mode invalido. Use um de: {SUPPORTED_MODES}")

    experiment_scope = str(payload.get("experiment_scope", "original")).strip().lower()
    if experiment_scope not in SUPPORTED_SCOPES:
        raise ValueError(f"experiment_scope invalido. Use um de: {SUPPORTED_SCOPES}")

    if mode in {"search_best", "search_and_run"} and experiment_scope != "original":
        raise ValueError(
            "Os modos de busca usam apenas dados originais. "
            "Defina experiment_scope='original'."
        )

    dataset_names = _parse_names(
        payload.get("dataset_names"),
        default=list(SUPPORTED_BASE_DATASETS),
    )
    invalid_datasets = sorted(set(dataset_names) - set(SUPPORTED_BASE_DATASETS))
    if invalid_datasets:
        raise ValueError(f"Datasets originais sem suporte: {invalid_datasets}")

    backbone_methods = _parse_names(
        payload.get("backbone_methods"),
        default=list(SUPPORTED_BACKBONE_METHODS),
    )
    invalid_methods = sorted(set(backbone_methods) - set(SUPPORTED_BACKBONE_METHODS))
    if invalid_methods:
        raise ValueError(f"Metodos backbone sem suporte: {invalid_methods}")

    backbone_dataset_names = _parse_names(payload.get("backbone_dataset_names"), default=[])
    model_names = _parse_names(payload.get("model_names"), default=list(SUPPORTED_MODELS))
    invalid_models = sorted(set(model_names) - set(SUPPORTED_MODELS))
    if invalid_models:
        raise ValueError(f"Modelos sem suporte: {invalid_models}")

    best_configs_file = payload.get("best_configs_file")
    resolved_best_configs_file = (
        _resolve_path(best_configs_file, base_dir=base_dir, default=base_dir / "missing.json")
        if best_configs_file
        else None
    )
    if mode == "run_best" and resolved_best_configs_file is None:
        raise ValueError("best_configs_file e obrigatorio quando mode='run_best'.")

    train_ratio = float(payload.get("train_ratio", 0.7))
    val_ratio = float(payload.get("val_ratio", 0.1))
    test_ratio = float(payload.get("test_ratio", 0.2))
    _validate_ratio_sum(train_ratio, val_ratio, test_ratio)

    run_label = str(payload.get("run_label", datetime.now().strftime("%d_%m_%Y-%Hh_%M"))).strip()
    if not run_label:
        raise ValueError("run_label nao pode ser vazio.")

    return PipelineConfig(
        config_file=path,
        mode=mode,
        experiment_scope=experiment_scope,
        dataset_names=dataset_names,
        backbone_dataset_names=backbone_dataset_names,
        backbone_methods=backbone_methods,
        backbone_alpha=float(payload.get("backbone_alpha", 0.4)),
        original_data_dir=_resolve_path(
            payload.get("original_data_dir"),
            base_dir=base_dir,
            default=PRIMARY_DATA_ROOT,
        ),
        backbone_data_dir=_resolve_path(
            payload.get("backbone_data_dir"),
            base_dir=base_dir,
            default=PRIMARY_DATA_ROOT,
        ),
        results_dir=_resolve_path(
            payload.get("results_dir"),
            base_dir=base_dir,
            default=WORKSPACE_ROOT / "outputs" / "forecasting",
        ),
        best_configs_file=resolved_best_configs_file,
        model_names=model_names,
        device=_resolve_device(payload.get("device")),
        seq_len=int(payload.get("seq_len", 12)),
        horizon=int(payload.get("horizon", 12)),
        batch_size=int(payload.get("batch_size", 32)),
        epochs=int(payload.get("epochs", 10)),
        seeds=_parse_seeds(payload.get("seeds")),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        normalize=_parse_bool(payload.get("normalize"), default=True),
        normalization_method=str(payload.get("normalization_method", "zscore")).strip().lower(),
        selection_metric=str(payload.get("selection_metric", "val_mae")).strip().lower(),
        generate_plots=_parse_bool(payload.get("generate_plots"), default=True),
        plots_num_nodes=int(payload.get("plots_num_nodes", 4)),
        plots_max_time_points=int(payload.get("plots_max_time_points", 350)),
        run_label=run_label,
        param_grids=payload.get("param_grids", {}),
        model_params=payload.get("model_params", {}),
    )

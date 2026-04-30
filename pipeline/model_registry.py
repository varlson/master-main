from __future__ import annotations

from copy import deepcopy
from typing import Any

from pipeline.bootstrap import ensure_workspace_root_on_path


SUPPORTED_MODELS = (
    "DCRNN",
    "GraphWaveNet",
    "MTGNN",
    "DGCRN",
    "STICformer",
    "PatchSTG",
)


def load_grid_search_registry() -> dict[str, Any]:
    ensure_workspace_root_on_path()
    from shared.MLFlow import (  # noqa: E402
        DCRNN_grid_search,
        DGCRN_grid_search,
        GraphWaveNet_grid_search,
        MTGNN_grid_search,
        PatchSTG_grid_search,
        STICformer_grid_search,
    )

    return {
        "DCRNN": DCRNN_grid_search,
        "GraphWaveNet": GraphWaveNet_grid_search,
        "MTGNN": MTGNN_grid_search,
        "DGCRN": DGCRN_grid_search,
        "STICformer": STICformer_grid_search,
        "PatchSTG": PatchSTG_grid_search,
    }


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def default_param_grids(seq_len: int, horizon: int, epochs: int) -> dict[str, dict[str, list[Any]]]:
    return {
        "DCRNN": {
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
        },
        "GraphWaveNet": {
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
        },
        "MTGNN": {
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
        },
        "DGCRN": {
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
        },
        "STICformer": {
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
        },
        "PatchSTG": {
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
        },
    }


def build_param_grids(
    *,
    seq_len: int,
    horizon: int,
    epochs: int,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, list[Any]]]:
    grids = deepcopy(default_param_grids(seq_len=seq_len, horizon=horizon, epochs=epochs))
    overrides = overrides or {}

    invalid_models = sorted(set(overrides) - set(SUPPORTED_MODELS))
    if invalid_models:
        raise ValueError(f"Modelos sem suporte em param_grids: {invalid_models}")

    for model_name, override_grid in overrides.items():
        if not isinstance(override_grid, dict):
            raise ValueError(f"param_grids.{model_name} deve ser um objeto JSON.")

        for param_name, param_value in override_grid.items():
            grids[model_name][param_name] = _as_list(param_value)

    return grids


def build_model_params(
    *,
    seq_len: int,
    horizon: int,
    epochs: int,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    default_grids = default_param_grids(seq_len=seq_len, horizon=horizon, epochs=epochs)
    resolved_params: dict[str, dict[str, Any]] = {}

    for model_name, grid in default_grids.items():
        resolved_params[model_name] = {
            param_name: values[0] if isinstance(values, list) and values else values
            for param_name, values in grid.items()
        }

    overrides = overrides or {}
    invalid_models = sorted(set(overrides) - set(SUPPORTED_MODELS))
    if invalid_models:
        raise ValueError(f"Modelos sem suporte em model_params: {invalid_models}")

    for model_name, override_params in overrides.items():
        if not isinstance(override_params, dict):
            raise ValueError(f"model_params.{model_name} deve ser um objeto JSON.")

        for param_name, param_value in override_params.items():
            if isinstance(param_value, list):
                raise ValueError(
                    f"model_params.{model_name}.{param_name} deve ser valor escalar, nao lista."
                )
            resolved_params[model_name][param_name] = param_value

    return resolved_params

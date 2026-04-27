from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_BASE_DATASETS = ("metr-la", "pems-bay")
SUPPORTED_BACKBONE_METHODS = ("disp_fil", "nois_corr", "high_sal", "doub_stoch")


@dataclass(frozen=True)
class DatasetGroup:
    experiment_type: str
    npy_dir: Path
    dataset_names: list[str]


def alpha_cut_name(alpha: float) -> str:
    return f"alpah_filter{str(alpha).replace('.', '_')}"


def infer_base_dataset_name(dataset_name: str) -> str | None:
    for name in SUPPORTED_BASE_DATASETS:
        if dataset_name == name or dataset_name.startswith(f"{name}-"):
            return name
    return None


def available_datasets(npy_dir: Path) -> list[str]:
    datasets = []
    for file in npy_dir.glob("*-h5.npy"):
        dataset = file.name.replace("-h5.npy", "")
        if (npy_dir / f"{dataset}-adj_mx.npy").exists():
            datasets.append(dataset)
    return sorted(set(datasets))


def resolve_dataset_paths(dataset_name: str, npy_dir: Path) -> tuple[Path, Path]:
    base_dataset_name = infer_base_dataset_name(dataset_name)
    base_data_path = npy_dir / f"{base_dataset_name}-h5.npy" if base_dataset_name else None

    data_path = npy_dir / f"{dataset_name}-h5.npy"
    adj_path = npy_dir / f"{dataset_name}-adj_mx.npy"

    if not data_path.exists() and base_data_path is not None:
        data_path = base_data_path

    if not data_path.exists() or not adj_path.exists():
        available = ", ".join(available_datasets(npy_dir)) or "(nenhum encontrado)"
        raise FileNotFoundError(
            f"Arquivos do dataset '{dataset_name}' nao encontrados em {npy_dir}. "
            f"Esperado: '{data_path.name}' e '{adj_path.name}'. Disponiveis: {available}"
        )

    return data_path, adj_path


def read_backbone_names_file(npy_dir: Path) -> list[str]:
    names_file = npy_dir / "backbone_data_names.txt"
    if not names_file.exists():
        return []

    with names_file.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def backbone_dataset_name(dataset_name: str, method: str, alpha: float) -> str:
    return f"{dataset_name}-by-{method}-with-{alpha_cut_name(alpha)}"


def method_matches_backbone_name(backbone_name: str, methods: list[str]) -> bool:
    return any(f"-by-{method}-with-" in backbone_name for method in methods)


def resolve_backbone_dataset_names(
    *,
    dataset_names: list[str],
    npy_dir: Path,
    methods: list[str],
    alpha: float,
    explicit_names: list[str] | None = None,
) -> list[str]:
    if explicit_names:
        return explicit_names

    generated_names = [
        backbone_dataset_name(dataset_name=dataset_name, method=method, alpha=alpha)
        for dataset_name in dataset_names
        for method in methods
    ]
    existing_generated_names = [
        name for name in generated_names if (npy_dir / f"{name}-adj_mx.npy").exists()
    ]
    if existing_generated_names:
        return list(dict.fromkeys(existing_generated_names))

    discovered_names: list[str] = []
    for dataset_name in dataset_names:
        prefix = f"{dataset_name}-by-"
        for backbone_name in read_backbone_names_file(npy_dir):
            if backbone_name.startswith(prefix) and method_matches_backbone_name(backbone_name, methods):
                discovered_names.append(backbone_name)

    if discovered_names:
        return list(dict.fromkeys(discovered_names))

    return list(dict.fromkeys(generated_names))

from __future__ import annotations

import argparse
from datetime import datetime
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.backbone.analysis import (
    DEFAULT_BACKBONE_ANALYSIS_ROOT,
    run_single_backbone_analysis,
)
from pipeline.bootstrap import WORKSPACE_ROOT, ensure_workspace_root_on_path

# if TYPE_CHECKING:
import networkx as nx
import numpy as np


DATA_DIR = WORKSPACE_ROOT / "data"
GRAPHML_DIR = DATA_DIR / "GraphML"
NPY_DIR = DATA_DIR / "npy"
PKL_DIR = DATA_DIR / "pkl"
DEFAULT_BACKBONE_OUTPUT_ROOT = WORKSPACE_ROOT / "outputs" / "backbone"
DEFAULT_DATASET_LIST = ["metr-la", "pems-bay", "wikivital-mathematics"]
DEFAULT_ALPHA = 0.3
DEFAULT_MIN_DEGREE = 1
SUPPORTED_METHODS = (
    "disp_fil",
    "nois_corr",
    "high_sal",
    "doub_stoch",
    "glanb",
    "h_backbone",
    "marg_likelihood",
)


def _dataset_backbone_combinations(
    methods: list[str], alpha: float
) -> list[tuple[str, str]]:
    alpha_cut = f"alpah_filter{str(alpha).replace('.', '_')}"
    return [(method, alpha_cut) for method in methods]


def _update_h5(
    original_h5: "np.ndarray", state_of_nodes: list[bool], output_name: str
) -> "np.ndarray":
    import numpy as np

    if len(state_of_nodes) != original_h5.shape[1]:
        raise ValueError(
            "Quantidade de flags em state_of_nodes difere do numero de nos "
            f"do H5: {len(state_of_nodes)} != {original_h5.shape[1]}"
        )

    keep_indices = [idx for idx, keep in enumerate(state_of_nodes) if keep]
    new_h5 = original_h5[:, keep_indices]
    output_file = NPY_DIR / f"{output_name}-h5.npy"
    np.save(output_file, new_h5)
    print(f"Updated H5 with shape {new_h5.shape} -> {output_file}")
    return new_h5


def _validate_inputs(datasets: list[str]) -> None:
    missing = []
    for dataset in datasets:
        pkl_path = PKL_DIR / f"{dataset}-adj_mx.pkl"
        h5_path = NPY_DIR / f"{dataset}-h5.npy"
        if not pkl_path.exists():
            missing.append(str(pkl_path))
        if not h5_path.exists():
            missing.append(str(h5_path))

    if missing:
        msg = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Arquivos necessarios nao encontrados:\n{msg}")


def _generate_graph_from_adjmx_nx(path: Path, name: str) -> "nx.Graph":
    import networkx as nx
    import numpy as np

    ext = path.suffix.lower()
    if ext == ".pkl":
        with path.open("rb") as file_obj:
            data = pickle.load(file_obj, encoding="latin1")
        if isinstance(data, (tuple, list)):
            _, _, adj = data
        else:
            adj = data
    elif ext == ".npy":
        adj = np.load(path, allow_pickle=True)
    elif ext == ".npz":
        npz = np.load(path, allow_pickle=True)
        if "adj" in npz:
            adj = npz["adj"]
        else:
            adj = None
            for value in npz.values():
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    adj = value
                    break
            if adj is None:
                raise ValueError("Arquivo .npz sem matriz de adjacencia 2D.")
    else:
        raise ValueError(f"Formato nao suportado: {ext}")

    adj = np.array(adj)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("A matriz de adjacencia deve ser quadrada.")

    graph = nx.from_numpy_array(adj)
    GRAPHML_DIR.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, GRAPHML_DIR / f"{name}.GraphML")
    return graph


def _save_filtered_graph(
    filtered_graph: "nx.Graph",
    filtered_graph_out_name: str,
    output_path: Path,
    weight: str = "weight",
) -> None:
    import networkx as nx

    adj_mx = nx.to_numpy_array(filtered_graph, weight=weight)
    print(f"Filtered adjacency matrix shape: {adj_mx.shape}")
    np.save(output_path, adj_mx)
    GRAPHML_DIR.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(filtered_graph, GRAPHML_DIR / f"{filtered_graph_out_name}.GraphML")


def _load_runtime_defaults() -> tuple[list[str], float, int]:
    datasets = list(DEFAULT_DATASET_LIST)
    alpha = DEFAULT_ALPHA
    min_degree = DEFAULT_MIN_DEGREE

    try:
        from config import ALPHA, DATASET_LIST, MIN_DEGREE

        datasets = list(DATASET_LIST)
        alpha = ALPHA
        min_degree = MIN_DEGREE
    except Exception as exc:
        print(f"Warning: could not import config.py ({exc}). Using local defaults.")

    return datasets, alpha, min_degree


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera backbones (adj + h5) como no notebook backbone/main.ipynb"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Lista de datasets (padrao: DATASET_LIST do config.py)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(SUPPORTED_METHODS),
        default=list(SUPPORTED_METHODS),
        help="Metodos de backbone a executar.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Threshold de alpha (padrao: ALPHA do config.py).",
    )
    parser.add_argument(
        "--min-degree",
        type=int,
        default=None,
        help="Grau minimo de no (padrao: MIN_DEGREE do config.py).",
    )
    return parser


def run_backbone_generation(
    *,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    alpha: float | None = None,
    min_degree: int | None = None,
    analysis_root: Path | None = None,
    manifest_output_path: Path | None = None,
) -> list[str]:
    import numpy as np

    from pipeline.backbone.filters import (
        DisparityFilter,
        DoublyStochasticFilter,
        GLANBFilter,
        HBackboneFilter,
        HighSalienceSkeleton,
        MarginalLikelihoodFilter,
        NoiseCorrectedFilter,
    )

    ensure_workspace_root_on_path()

    defaults_dataset, defaults_alpha, defaults_min_degree = _load_runtime_defaults()

    resolved_datasets = datasets if datasets is not None else defaults_dataset
    resolved_methods = methods if methods is not None else list(SUPPORTED_METHODS)
    resolved_alpha = defaults_alpha if alpha is None else alpha
    resolved_min_degree = defaults_min_degree if min_degree is None else min_degree
    resolved_analysis_root = (
        analysis_root
        if analysis_root is not None
        else DEFAULT_BACKBONE_ANALYSIS_ROOT
    )
    resolved_manifest_output_path = (
        manifest_output_path
        if manifest_output_path is not None
        else DEFAULT_BACKBONE_OUTPUT_ROOT
        / "manifests"
        / f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    NPY_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHML_DIR.mkdir(parents=True, exist_ok=True)

    _validate_inputs(resolved_datasets)

    combinations = _dataset_backbone_combinations(
        methods=resolved_methods,
        alpha=resolved_alpha,
    )
    if not combinations:
        raise ValueError("Nenhuma combinacao para executar. Revise --methods.")

    backbone_data_names: list[str] = []

    for dataset in resolved_datasets:
        graph_path = PKL_DIR / f"{dataset}-adj_mx.pkl"
        h5_path = NPY_DIR / f"{dataset}-h5.npy"

        print(f"\nDataset: {dataset}")
        print(f"  graph: {graph_path}")
        print(f"  h5:    {h5_path}")

        graph = _generate_graph_from_adjmx_nx(graph_path, dataset)
        h5_matrix = np.load(h5_path)

        for method, cut in combinations:
            output_name = f"{dataset}-by-{method}-with-{cut}".strip()
            backbone_data_names.append(output_name)
            print(f"Processing {output_name}...")

            if method == "disp_fil":
                filter_obj = DisparityFilter(graph)
            elif method == "nois_corr":
                filter_obj = NoiseCorrectedFilter(
                    graph, undirected=True, use_p_value=False
                )
            elif method == "high_sal":
                filter_obj = HighSalienceSkeleton(graph)
            elif method == "doub_stoch":
                filter_obj = DoublyStochasticFilter(graph)
            elif method == "glanb":
                filter_obj = GLANBFilter(graph)
            elif method == "h_backbone":
                filter_obj = HBackboneFilter(graph)
            elif method == "marg_likelihood":
                filter_obj = MarginalLikelihoodFilter(graph)
            else:
                raise ValueError(f"Metodo de backbone nao suportado: {method}")

            filtered_graph = filter_obj.filter_by_alpha(
                alpha=resolved_alpha,
                min_degree=resolved_min_degree,
            )
            nodes_to_keep = filter_obj.nodesToKeep

            if filtered_graph.number_of_nodes() != graph.number_of_nodes():
                _update_h5(h5_matrix, nodes_to_keep, output_name)
            else:
                print("No removals in nodes; H5 not updated.")

            _save_filtered_graph(
                filtered_graph,
                output_name,
                NPY_DIR / f"{output_name}-adj_mx.npy",
            )

            analysis_output_root = resolved_analysis_root / dataset / output_name
            backbone_graphml_path = GRAPHML_DIR / f"{output_name}.GraphML"
            original_graphml_path = GRAPHML_DIR / f"{dataset}.GraphML"
            analysis_output = run_single_backbone_analysis(
                dataset=dataset,
                original_path=original_graphml_path,
                backbone_path=backbone_graphml_path,
                output_root=analysis_output_root,
            )
            print(f"Analysis report saved to {analysis_output}")

    names_file = NPY_DIR / "backbone_data_names.txt"
    with names_file.open("w", encoding="utf-8") as file_obj:
        for name in backbone_data_names:
            file_obj.write(f"{name}\n")

    print(f"\nBackbone names saved to {names_file}")

    resolved_manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "datasets": resolved_datasets,
        "methods": resolved_methods,
        "alpha": resolved_alpha,
        "min_degree": resolved_min_degree,
        "analysis_root": str(resolved_analysis_root),
        "generated_backbones": backbone_data_names,
    }
    with resolved_manifest_output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(manifest_payload, file_obj, indent=2, ensure_ascii=False)
    print(f"Generation manifest saved to {resolved_manifest_output_path}")

    print("Done.")
    return backbone_data_names


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_backbone_generation(
        datasets=args.datasets,
        methods=args.methods,
        alpha=args.alpha,
        min_degree=args.min_degree,
    )

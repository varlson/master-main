#!/usr/bin/env python3
# encoding: utf-8

"""
Script executavel equivalente ao notebook backbone/main.ipynb.

Uso:
    cd backbone
    python3 main.py

Tambem aceita parametros opcionais:
    python3 main.py --datasets metr-la --methods disp_fil high_sal doub_stoch glanb
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
NPY_DIR = DATA_DIR / "npy"
PKL_DIR = DATA_DIR / "pkl"
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



def _prepare_imports() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


def _dataset_backbone_combinations(
    methods: list[str], alpha: float
) -> list[tuple[str, str]]:
    alpha_cut = f"alpah_filter{str(alpha).replace('.', '_')}"
    return [(method, alpha_cut) for method in methods]


def _update_h5(
    original_h5: np.ndarray, state_of_nodes: list[bool], output_name: str
) -> np.ndarray:
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


def _generate_graph_from_adjmx_nx(path: Path, name: str) -> nx.Graph:
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
    nx.write_graphml(graph, DATA_DIR / "GraphML" / f"{name}.GraphML")
    return graph


def _save_filtered_graph(
    original_graph: nx.Graph,
    filtered_graph: nx.Graph,
    filtered_graph_out_name: str,
    output_path: Path,
    weight: str = "weight",
) -> None:
    _ = original_graph
    adj_mx = nx.to_numpy_array(filtered_graph, weight=weight)
    print(f"Filtered adjacency matrix shape: {adj_mx.shape}")
    np.save(output_path, adj_mx)
    nx.write_graphml(
        filtered_graph, DATA_DIR / "GraphML" / f"{filtered_graph_out_name}.GraphML"
    )


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


def main() -> None:
    os.chdir(SCRIPT_DIR)
    _prepare_imports()

    from analisys import run_single_backbone_analysis
    from disparity_filter import DisparityFilter
    from doubly_stochastic_filter import DoublyStochasticFilter
    from glanb import GLANBFilter
    from h_backbone import HBackboneFilter
    from high_salience_skeleton import HighSalienceSkeleton
    from marginal_likelihood import MarginalLikelihoodFilter
    from noise_corrected import NoiseCorrectedFilter

    parser = build_parser()
    args = parser.parse_args()

    defaults_dataset, defaults_alpha, defaults_min_degree = _load_runtime_defaults()

    datasets = args.datasets if args.datasets is not None else defaults_dataset
    alpha = defaults_alpha if args.alpha is None else args.alpha
    min_degree = defaults_min_degree if args.min_degree is None else args.min_degree

    NPY_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "GraphML").mkdir(parents=True, exist_ok=True)

    _validate_inputs(datasets)

    combinations = _dataset_backbone_combinations(
        methods=args.methods,
        alpha=alpha,
    )
    if not combinations:
        raise ValueError("Nenhuma combinacao para executar. Revise --methods.")

    backbone_data_names: list[str] = []

    for dataset in datasets:
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
                df = DisparityFilter(graph)
                filtered_graph = df.filter_by_alpha(
                    alpha=alpha, min_degree=min_degree
                )
                nodes_to_keep = df.nodesToKeep
            elif method == "nois_corr":
                ncf = NoiseCorrectedFilter(graph, undirected=True, use_p_value=False)
                filtered_graph = ncf.filter_by_alpha(
                    alpha=alpha, min_degree=min_degree
                )
                nodes_to_keep = ncf.nodesToKeep
            elif method == "high_sal":
                hss = HighSalienceSkeleton(graph)
                filtered_graph = hss.filter_by_alpha(
                    alpha=alpha, min_degree=min_degree
                )
                nodes_to_keep = hss.nodesToKeep
            elif method == "doub_stoch":
                dsf = DoublyStochasticFilter(graph)
                filtered_graph = dsf.filter_by_alpha(
                    alpha=alpha, min_degree=min_degree
                )
                nodes_to_keep = dsf.nodesToKeep
            elif method == "glanb":
                glb = GLANBFilter(graph)
                filtered_graph = glb.filter_by_alpha(
                    alpha=alpha, min_degree=min_degree
                )
                nodes_to_keep = glb.nodesToKeep
            elif method == "h_backbone":
                hbf = HBackboneFilter(graph)
                filtered_graph = hbf.filter_by_alpha(
                    alpha=alpha, min_degree=min_degree
                )
                nodes_to_keep = hbf.nodesToKeep
            elif method == "marg_likelihood":
                mlf = MarginalLikelihoodFilter(graph)
                filtered_graph = mlf.filter_by_alpha(
                    alpha=alpha, min_degree=min_degree
                )
                nodes_to_keep = mlf.nodesToKeep
            else:
                raise ValueError(f"Metodo de backbone nao suportado: {method}")

            if filtered_graph.number_of_nodes() != graph.number_of_nodes():
                _update_h5(h5_matrix, nodes_to_keep, output_name)
            else:
                print("No removals in nodes; H5 not updated.")

            _save_filtered_graph(
                graph,
                filtered_graph,
                output_name,
                NPY_DIR / f"{output_name}-adj_mx.npy",
            )

            analysis_output_root = SCRIPT_DIR / "analisys" / dataset / output_name
            backbone_graphml_path = DATA_DIR / "GraphML" / f"{output_name}.GraphML"
            original_graphml_path = DATA_DIR / "GraphML" / f"{dataset}.GraphML"
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
    print("Done.")


if __name__ == "__main__":
    main()

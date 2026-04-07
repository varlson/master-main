#!/usr/bin/env python3
# encoding: utf-8

"""
Script executavel equivalente ao notebook backbone/main.ipynb.

Uso:
    cd backbone
    python3 main.py

Tambem aceita parametros opcionais:
    python3 main.py --datasets metr-la --methods disp_fil high_sal --cuts alpha
"""

from __future__ import annotations

import argparse
import itertools
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
DEFAULT_DATASET_LIST = ["metr-la", "pems-bay"]
DEFAULT_ALPHA = 0.1
DEFAULT_PERCENTILE = 0.30
DEFAULT_MIN_DEGREE = 1


def _prepare_imports() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))


def _dataset_backbone_combinations(
    methods: list[str], alpha: float, percentile: float, cuts: list[str]
) -> list[tuple[str, str]]:
    alpha_cut = f"alpah_filter{str(alpha).replace('.', '_')}"
    percentile_cut = f"percen_filter{str(percentile).replace('.', '_')}"

    active_cuts: list[str] = []
    if "alpha" in cuts:
        active_cuts.append(alpha_cut)
    if "percentile" in cuts:
        active_cuts.append(percentile_cut)

    return list(itertools.product(methods, active_cuts))


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


def _load_runtime_defaults() -> tuple[list[str], float, float, int]:
    datasets = list(DEFAULT_DATASET_LIST)
    alpha = DEFAULT_ALPHA
    percentile = DEFAULT_PERCENTILE
    min_degree = DEFAULT_MIN_DEGREE

    try:
        from config import ALPHA, DATASET_LIST, MIN_DEGREE, PERCENTILE

        datasets = list(DATASET_LIST)
        alpha = ALPHA
        percentile = PERCENTILE
        min_degree = MIN_DEGREE
    except Exception as exc:
        print(f"Warning: could not import config.py ({exc}). Using local defaults.")

    return datasets, alpha, percentile, min_degree


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
        choices=["disp_fil", "nois_corr", "high_sal"],
        default=["disp_fil", "nois_corr"],
        help="Metodos de backbone a executar.",
    )
    parser.add_argument(
        "--cuts",
        nargs="+",
        choices=["alpha", "percentile"],
        default=["alpha", "percentile"],
        help="Tipos de corte a executar.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Threshold de alpha (padrao: ALPHA do config.py).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Threshold de percentil (padrao: PERCENTILE do config.py).",
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

    from disparity_filter import DisparityFilter
    from high_salience_skeleton import HighSalienceSkeleton
    from noise_corrected import NoiseCorrectedFilter

    parser = build_parser()
    args = parser.parse_args()

    defaults_dataset, defaults_alpha, defaults_percentile, defaults_min_degree = (
        _load_runtime_defaults()
    )

    datasets = args.datasets if args.datasets is not None else defaults_dataset
    alpha = defaults_alpha if args.alpha is None else args.alpha
    percentile = defaults_percentile if args.percentile is None else args.percentile
    min_degree = defaults_min_degree if args.min_degree is None else args.min_degree

    NPY_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "GraphML").mkdir(parents=True, exist_ok=True)

    _validate_inputs(datasets)

    combinations = _dataset_backbone_combinations(
        methods=args.methods,
        alpha=alpha,
        percentile=percentile,
        cuts=args.cuts,
    )
    if not combinations:
        raise ValueError("Nenhuma combinacao para executar. Revise --methods e --cuts.")

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
                if cut.startswith("percen"):
                    filtered_graph = df.filter_by_percentile(
                        percentile=percentile, min_degree=min_degree
                    )
                else:
                    filtered_graph = df.filter_by_alpha(
                        alpha=alpha, min_degree=min_degree
                    )
                nodes_to_keep = df.nodesToKeep
            elif method == "nois_corr":
                ncf = NoiseCorrectedFilter(graph, undirected=True, use_p_value=False)
                if cut.startswith("percen"):
                    filtered_graph = ncf.filter_by_percentile(
                        percentile=percentile, min_degree=min_degree
                    )
                else:
                    filtered_graph = ncf.filter_by_alpha(
                        alpha=alpha, min_degree=min_degree
                    )
                nodes_to_keep = ncf.nodesToKeep
            elif method == "high_sal":
                hss = HighSalienceSkeleton(graph)
                if cut.startswith("percen"):
                    filtered_graph = hss.filter_by_percentile(
                        percentile=percentile, min_degree=min_degree
                    )
                else:
                    filtered_graph = hss.filter_by_alpha(
                        alpha=alpha, min_degree=min_degree
                    )
                nodes_to_keep = hss.nodesToKeep
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

    names_file = NPY_DIR / "backbone_data_names.txt"
    with names_file.open("w", encoding="utf-8") as file_obj:
        for name in backbone_data_names:
            file_obj.write(f"{name}\n")

    print(f"\nBackbone names saved to {names_file}")
    print("Done.")


if __name__ == "__main__":
    main()

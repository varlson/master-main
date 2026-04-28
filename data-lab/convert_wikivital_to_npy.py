#!/usr/bin/env python3
"""
Converte um arquivo WikiVital em JSON para o formato .npy esperado pelo pipeline:

- <dataset>-h5.npy: serie temporal com shape (T, N)
- <dataset>-adj_mx.npy: matriz de adjacencia com shape (N, N)

Tambem salva um arquivo de metadata para preservar o mapeamento de nos e datas.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte um dataset WikiVital JSON para serie temporal e adjacencia em .npy."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data-lab/wikivital_mathematics.json"),
        help="Caminho do arquivo JSON de entrada.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/npy"),
        help="Diretorio onde os arquivos .npy serao salvos.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikivital-mathematics",
        help="Prefixo usado nos arquivos de saida.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help="Tipo numerico usado nos arrays salvos.",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Espelha a adjacencia para forcar um grafo nao direcionado.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Nao salva o arquivo auxiliar de metadata.",
    )
    parser.add_argument(
        "--no-pkl",
        action="store_true",
        help="Nao salva o arquivo .pkl auxiliar para o pipeline de backbone.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"O JSON raiz deve ser um objeto. Recebido: {type(payload).__name__}")

    return payload


def extract_time_keys(payload: dict) -> list[str]:
    time_keys = [
        key
        for key, value in payload.items()
        if isinstance(value, dict) and "y" in value
    ]
    if not time_keys:
        raise ValueError("Nenhuma entrada temporal com campo 'y' foi encontrada no JSON.")
    return sorted(time_keys, key=lambda key: int(key))


def build_timeseries(payload: dict, time_keys: list[str], dtype: np.dtype) -> tuple[np.ndarray, list[dict]]:
    rows: list[list[float]] = []
    time_index: list[dict] = []
    expected_num_nodes: int | None = None

    for key in time_keys:
        step = payload[key]
        y = step.get("y")
        if not isinstance(y, list):
            raise ValueError(f"A entrada temporal '{key}' nao possui uma lista valida em 'y'.")

        if expected_num_nodes is None:
            expected_num_nodes = len(y)
        elif len(y) != expected_num_nodes:
            raise ValueError(
                f"Inconsistencia em '{key}': esperado vetor 'y' com {expected_num_nodes} valores, "
                f"mas foram encontrados {len(y)}."
            )

        rows.append(y)
        time_index.append(
            {
                "step": int(key),
                "index": step.get("index"),
                "year": step.get("year"),
                "month": step.get("month"),
                "day": step.get("day"),
            }
        )

    return np.asarray(rows, dtype=dtype), time_index


def build_adjacency(
    payload: dict,
    num_nodes: int,
    dtype: np.dtype,
    undirected: bool,
) -> np.ndarray:
    edges = payload.get("edges")
    weights = payload.get("weights")

    if not isinstance(edges, list):
        raise ValueError("Campo 'edges' ausente ou invalido.")
    if weights is None:
        weights = [1.0] * len(edges)
    if not isinstance(weights, list):
        raise ValueError("Campo 'weights' invalido; esperado uma lista.")
    if len(edges) != len(weights):
        raise ValueError(
            f"Quantidade de arestas e pesos diferente: {len(edges)} arestas vs {len(weights)} pesos."
        )

    adj = np.zeros((num_nodes, num_nodes), dtype=dtype)

    for idx, (edge, weight) in enumerate(zip(edges, weights)):
        if not isinstance(edge, list) or len(edge) != 2:
            raise ValueError(f"Aresta invalida no indice {idx}: {edge!r}")

        src, dst = edge
        if not isinstance(src, int) or not isinstance(dst, int):
            raise ValueError(f"Aresta no indice {idx} deve conter inteiros: {edge!r}")
        if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            raise ValueError(
                f"Aresta fora do intervalo valido no indice {idx}: {edge!r} para {num_nodes} nos."
            )

        adj[src, dst] = weight
        if undirected:
            adj[dst, src] = weight

    return adj


def build_node_mapping(payload: dict, num_nodes: int) -> list[str | None]:
    node_ids = payload.get("node_ids")
    if not isinstance(node_ids, dict):
        return [None] * num_nodes

    index_to_title: list[str | None] = [None] * num_nodes
    for title, index in node_ids.items():
        if not isinstance(index, int):
            raise ValueError(f"Indice invalido em node_ids para '{title}': {index!r}")
        if index < 0 or index >= num_nodes:
            raise ValueError(
                f"Indice fora do intervalo em node_ids para '{title}': {index} (num_nodes={num_nodes})"
            )
        index_to_title[index] = title
    return index_to_title


def save_metadata(
    output_path: Path,
    dataset_name: str,
    num_nodes: int,
    num_timesteps: int,
    time_index: list[dict],
    index_to_title: list[str | None],
    undirected: bool,
) -> None:
    metadata = {
        "dataset_name": dataset_name,
        "num_nodes": num_nodes,
        "num_timesteps": num_timesteps,
        "adjacency_mode": "undirected" if undirected else "directed",
        "time_index": time_index,
        "index_to_title": index_to_title,
    }
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_pkl_payload(index_to_title: list[str | None], adjacency: np.ndarray) -> list[object]:
    # Mantem o mesmo padrao usado comumente em datasets de trafego:
    # [lista_de_node_ids, dict_node_id_para_indice, adj_mx]
    ordered_node_ids: list[str] = []
    node_id_to_ind: dict[str, int] = {}

    for index, title in enumerate(index_to_title):
        node_id = title if title is not None else str(index)
        ordered_node_ids.append(node_id)
        node_id_to_ind[node_id] = index

    return [ordered_node_ids, node_id_to_ind, adjacency]


def main() -> None:
    args = parse_args()
    dtype = np.float32 if args.dtype == "float32" else np.float64

    payload = load_payload(args.input)
    time_keys = extract_time_keys(payload)
    timeseries, time_index = build_timeseries(payload, time_keys, dtype)
    num_timesteps, num_nodes = timeseries.shape
    adjacency = build_adjacency(payload, num_nodes, dtype, args.undirected)
    index_to_title = build_node_mapping(payload, num_nodes)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    h5_path = args.output_dir / f"{args.dataset_name}-h5.npy"
    adj_path = args.output_dir / f"{args.dataset_name}-adj_mx.npy"
    pkl_path = args.output_dir.parent / "pkl" / f"{args.dataset_name}-adj_mx.pkl"
    metadata_path = args.output_dir / f"{args.dataset_name}-metadata.json"

    np.save(h5_path, timeseries)
    np.save(adj_path, adjacency)

    if not args.no_pkl:
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        pkl_payload = build_pkl_payload(index_to_title, adjacency)
        with pkl_path.open("wb") as file_obj:
            pickle.dump(pkl_payload, file_obj, protocol=2)

    if not args.no_metadata:
        save_metadata(
            output_path=metadata_path,
            dataset_name=args.dataset_name,
            num_nodes=num_nodes,
            num_timesteps=num_timesteps,
            time_index=time_index,
            index_to_title=index_to_title,
            undirected=args.undirected,
        )

    print(f"Serie temporal salva em: {h5_path}")
    print(f"Matriz de adjacencia salva em: {adj_path}")
    print(f"Shape da serie temporal: {timeseries.shape}")
    print(f"Shape da adjacencia: {adjacency.shape}")
    print(f"Grafo tratado como: {'nao direcionado' if args.undirected else 'direcionado'}")
    if not args.no_pkl:
        print(f"PKL de adjacencia salvo em: {pkl_path}")
    if not args.no_metadata:
        print(f"Metadata salva em: {metadata_path}")


if __name__ == "__main__":
    main()

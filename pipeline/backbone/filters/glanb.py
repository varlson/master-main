#!/usr/bin/env python
# encoding: utf-8

"""
GLANB para extração de backbone de redes complexas.

Esta implementação adapta o algoritmo para a mesma interface usada pelos
demais filtros do projeto: `compute_filter()`, `filter_by_alpha()`,
`filter_by_percentile()`, `get_edges_dataframe()` e `nodesToKeep`.

Nota metodológica:
O GLANB produz uma medida `SI` por aresta, onde valores menores indicam
maior relevância estrutural. Para uniformizar o pipeline, definimos:

    alpha = SI

Assim, como nos demais filtros:
    - alpha baixo -> aresta relevante -> preservar
    - alpha alto  -> aresta pouco relevante -> remover
"""

from __future__ import annotations

from typing import Dict, Hashable, Optional, Tuple

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


def count_included_subarrays(arrays, target_array) -> int:
    count = 0
    target_len = len(target_array)
    for array in arrays:
        array_len = len(array)
        for i in range(array_len - target_len + 1):
            if array[i : i + target_len] == target_array:
                count += 1
    return count


class GLANBFilter:
    """
    Adaptação orientada a classe do algoritmo GLANB.

    O método usa caminhos mínimos ponderados para medir a importância de uma
    aresta em relação às árvores de caminhos mínimos originadas em cada nó.
    """

    METHOD_NAME: str = "glanb"

    def __init__(self, graph: nx.Graph, c: float = 1.0, epsilon: float = 1e-12):
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.c = c
        self.epsilon = epsilon
        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.si_measures: list[float] = []
        self.alpha_measures: list[float] = []
        self._filter_applied: bool = False
        self.nodesToKeep: list = []
        self._is_directed = self.graph.is_directed()

    def _edge_key(self, source: Hashable, target: Hashable):
        if self._is_directed:
            return (source, target)
        return frozenset((source, target))

    def _distance_from_weight(self, weight: float) -> float:
        safe_weight = max(float(weight), self.epsilon)
        return 1.0 / safe_weight

    def compute_filter(self) -> pd.DataFrame:
        """
        Calcula a medida SI do GLANB para cada aresta do grafo.

        Returns:
            DataFrame com métricas das arestas e coluna `alpha = SI`.
        """
        graph = self.graph.copy()

        if graph.number_of_edges() == 0:
            self.edges_df = pd.DataFrame(
                columns=[
                    "source",
                    "target",
                    "weight",
                    "distance",
                    "best_source",
                    "degree_source",
                    "g_ij",
                    "g_is",
                    "I_ij",
                    "SI",
                    "alpha",
                    "alpha_percentile",
                    "SI_percentile",
                ]
            )
            self._create_nodes_dataframe()
            self._filter_applied = True
            return self.edges_df

        for u, v, data in graph.edges(data=True):
            data["distance"] = self._distance_from_weight(data.get("weight", 1.0))

        igraph_graph = ig.Graph.TupleList(
            (
                (u, v, data["distance"])
                for u, v, data in graph.edges(data=True)
            ),
            directed=self._is_directed,
            weights=True,
        )
        node_to_index = {
            vertex["name"]: vertex.index for vertex in igraph_graph.vs
        }

        best_metrics: Dict[Hashable, Dict[str, float | int | Hashable]] = {}

        for source in graph.nodes():
            degree_source = graph.degree(source)
            if degree_source <= 1:
                continue

            source_index = node_to_index[source]
            shortest_paths = igraph_graph.get_all_shortest_paths(
                source_index,
                weights="weight",
            )
            g_is = len(shortest_paths) - 1
            if g_is <= 0:
                continue

            for u, v in graph.edges(source):
                g_ij = count_included_subarrays(
                    shortest_paths,
                    [node_to_index[u], node_to_index[v]],
                )
                I_ij = g_ij / g_is
                SI = (1.0 - I_ij) ** ((degree_source - 1) ** self.c)

                edge_key = self._edge_key(u, v)
                current = best_metrics.get(edge_key)
                if current is None or SI < float(current["SI"]):
                    best_metrics[edge_key] = {
                        "best_source": source,
                        "degree_source": degree_source,
                        "g_ij": g_ij,
                        "g_is": g_is,
                        "I_ij": I_ij,
                        "SI": SI,
                    }

        edges_data = []
        self.si_measures = []
        self.alpha_measures = []

        for u, v, edge_data in graph.edges(data=True):
            edge_key = self._edge_key(u, v)
            metrics = best_metrics.get(
                edge_key,
                {
                    "best_source": None,
                    "degree_source": 0,
                    "g_ij": 0,
                    "g_is": 0,
                    "I_ij": 0.0,
                    "SI": 1.0,
                },
            )
            si_value = float(metrics["SI"])
            edge_info = {
                "source": u,
                "target": v,
                "weight": float(edge_data.get("weight", 1.0)),
                "distance": float(edge_data["distance"]),
                "best_source": metrics["best_source"],
                "degree_source": int(metrics["degree_source"]),
                "g_ij": int(metrics["g_ij"]),
                "g_is": int(metrics["g_is"]),
                "I_ij": float(metrics["I_ij"]),
                "SI": si_value,
                "alpha": si_value,
            }

            if "pos" in graph.nodes[u]:
                edge_info["source_x"] = graph.nodes[u]["pos"][0]
                edge_info["source_y"] = graph.nodes[u]["pos"][1]
            if "pos" in graph.nodes[v]:
                edge_info["target_x"] = graph.nodes[v]["pos"][0]
                edge_info["target_y"] = graph.nodes[v]["pos"][1]

            edges_data.append(edge_info)
            self.si_measures.append(si_value)
            self.alpha_measures.append(si_value)

        self.edges_df = pd.DataFrame(edges_data)
        if len(self.edges_df) > 0:
            alpha_scores = self.edges_df["alpha"].tolist()
            self.edges_df["alpha_percentile"] = self.edges_df["alpha"].apply(
                lambda value: percentileofscore(alpha_scores, value) / 100.0
            )
            self.edges_df["SI_percentile"] = self.edges_df["SI"].apply(
                lambda value: percentileofscore(alpha_scores, value) / 100.0
            )
        else:
            self.edges_df["alpha_percentile"] = 0.0
            self.edges_df["SI_percentile"] = 0.0

        self._create_nodes_dataframe()
        self._filter_applied = True
        return self.edges_df

    def filter_by_alpha(self, alpha: float, min_degree: int = 1) -> nx.Graph:
        if not self._filter_applied:
            self.compute_filter()

        filtered_graph = self.graph.copy()
        edges_to_remove = [
            (row["source"], row["target"])
            for _, row in self.edges_df.iterrows()
            if row["alpha"] >= alpha
        ]
        filtered_graph.remove_edges_from(edges_to_remove)

        self.nodesToKeep, nodes_to_remove = self._classify_nodes(
            filtered_graph,
            min_degree,
        )
        filtered_graph.remove_nodes_from(nodes_to_remove)
        return filtered_graph

    def filter_by_percentile(self, percentile: float, min_degree: int = 1) -> nx.Graph:
        if not self._filter_applied:
            self.compute_filter()

        filtered_graph = self.graph.copy()
        edges_to_remove = [
            (row["source"], row["target"])
            for _, row in self.edges_df.iterrows()
            if row["alpha_percentile"] > percentile
        ]
        filtered_graph.remove_edges_from(edges_to_remove)

        self.nodesToKeep, nodes_to_remove = self._classify_nodes(
            filtered_graph,
            min_degree,
        )
        filtered_graph.remove_nodes_from(nodes_to_remove)
        return filtered_graph

    def _classify_nodes(
        self,
        filtered_graph: nx.Graph,
        min_degree: int,
    ) -> Tuple[list, list]:
        nodes_to_keep_flags = []
        nodes_to_remove = []

        for node in self.graph.nodes():
            if node in filtered_graph and filtered_graph.degree(node) >= min_degree:
                nodes_to_keep_flags.append(True)
            else:
                nodes_to_keep_flags.append(False)
                if node in filtered_graph:
                    nodes_to_remove.append(node)

        return nodes_to_keep_flags, nodes_to_remove

    def _create_nodes_dataframe(self) -> pd.DataFrame:
        nodes_data = []
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            node_info = {
                "node_id": node_id,
                "degree": self.graph.degree(node_id),
            }
            if "pos" in node_data:
                node_info["x"] = node_data["pos"][0]
                node_info["y"] = node_data["pos"][1]
            if "label" in node_data:
                node_info["label"] = node_data["label"]
            for key, value in node_data.items():
                if key not in ("pos", "label"):
                    node_info[key] = value
            nodes_data.append(node_info)

        self.nodes_df = pd.DataFrame(nodes_data)
        return self.nodes_df

    def get_edges_dataframe(self) -> pd.DataFrame:
        if not self._filter_applied:
            self.compute_filter()
        return self.edges_df

    def get_nodes_dataframe(self) -> pd.DataFrame:
        if self.nodes_df is None:
            self._create_nodes_dataframe()
        return self.nodes_df

    def get_summary_statistics(self) -> Dict:
        if not self._filter_applied:
            self.compute_filter()

        if not self.si_measures:
            return {
                "num_nodes": len(self.graph.nodes()),
                "num_edges": len(self.graph.edges()),
                "si_mean": 0.0,
                "si_median": 0.0,
                "si_std": 0.0,
                "si_min": 0.0,
                "si_max": 0.0,
            }

        return {
            "num_nodes": len(self.graph.nodes()),
            "num_edges": len(self.graph.edges()),
            "si_mean": float(np.mean(self.si_measures)),
            "si_median": float(np.median(self.si_measures)),
            "si_std": float(np.std(self.si_measures)),
            "si_min": float(np.min(self.si_measures)),
            "si_max": float(np.max(self.si_measures)),
        }

    def print_quantiles(self, num_quantiles: int = 10):
        if not self._filter_applied:
            self.compute_filter()

        bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
        quantiles = self.edges_df["alpha"].quantile(bins)

        print("\n" + "=" * 70)
        print("Quantis do GLANB")
        print("=" * 70)
        print(f"\n{'Percentil':<15} {'SI/Alpha':<15} {'Interpretação'}")
        print("-" * 70)
        for percentile, value in quantiles.items():
            if value < 0.2:
                interpretation = "Muito relevante"
            elif value < 0.5:
                interpretation = "Relevante"
            else:
                interpretation = "Pouco relevante"
            print(f"{percentile:>6.2%} {value:>15.4f} {interpretation}")
        print("=" * 70)


def glanb(data, c: float = 1.0):
    """
    Wrapper legado para manter compatibilidade com o uso antigo.

    Returns:
        Grafo NetworkX anotado com `SI` e `alpha`.
    """
    if isinstance(data, pd.DataFrame):
        graph = nx.from_pandas_edgelist(
            data,
            source="source",
            target="target",
            edge_attr="weight",
            create_using=nx.Graph(),
        )
    elif isinstance(data, nx.Graph):
        graph = data.copy()
    else:
        raise TypeError("data should be a pandas dataframe or nx graph")

    glanb_filter = GLANBFilter(graph, c=c)
    edges_df = glanb_filter.compute_filter()

    annotated_graph = graph.copy()
    for _, row in edges_df.iterrows():
        annotated_graph[row["source"]][row["target"]].update(
            {
                "SI": row["SI"],
                "alpha": row["alpha"],
            }
        )
    return annotated_graph

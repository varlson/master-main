#!/usr/bin/env python
# encoding: utf-8

"""
H-Backbone para extração de backbone de redes complexas.

Esta implementação padroniza o algoritmo para a mesma interface usada pelos
demais métodos do projeto.

Critério natural do método:
    uma aresta pertence ao backbone se:

        h_bridge(edge) >= h_bridge_threshold
        OU
        weight(edge)   >= h_weight_threshold

Para uniformizar a interface do pipeline, definimos também um score contínuo:

    score = max(weight / h_weight_threshold, h_bridge / h_bridge_threshold)

Depois normalizamos esse score para [0, 1] e usamos:

    alpha = 1 - score_normalized

Assim:
    - alpha baixo -> aresta relevante -> preservar
    - alpha alto  -> aresta pouco relevante -> remover
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


class HBackboneFilter:
    METHOD_NAME: str = "h_backbone"

    def __init__(self, graph: nx.Graph):
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.score_measures: list[float] = []
        self.alpha_measures: list[float] = []
        self._filter_applied: bool = False
        self.nodesToKeep: list = []
        self.h_weight_threshold: float = 0.0
        self.h_bridge_threshold: float = 0.0

    @staticmethod
    def _compute_h_index(values: list[float]) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)
        result = 0.0
        for index, cited in enumerate(ordered):
            result = len(ordered) - index
            if result <= cited:
                break
        return float(result)

    def compute_filter(self) -> pd.DataFrame:
        if self.graph.number_of_edges() == 0:
            self.edges_df = pd.DataFrame(
                columns=[
                    "source",
                    "target",
                    "weight",
                    "h_bridge",
                    "weight_ratio",
                    "bridge_ratio",
                    "score_raw",
                    "score",
                    "alpha",
                    "in_backbone",
                    "alpha_percentile",
                    "score_percentile",
                ]
            )
            self._create_nodes_dataframe()
            self._filter_applied = True
            return self.edges_df

        betweenness_values = nx.edge_betweenness_centrality(
            self.graph,
            weight="weight",
            normalized=False,
        )

        num_nodes = max(len(self.graph.nodes()), 1)
        h_bridge_map = {
            edge: round(betweenness_values[edge] / num_nodes, 3)
            for edge in betweenness_values
        }
        nx.set_edge_attributes(
            self.graph,
            {edge: {"h_bridge": value} for edge, value in h_bridge_map.items()},
        )

        weight_values = [
            float(data.get("weight", 1.0))
            for _, _, data in self.graph.edges(data=True)
        ]
        bridge_values = list(h_bridge_map.values())

        self.h_weight_threshold = self._compute_h_index(weight_values)
        self.h_bridge_threshold = self._compute_h_index(bridge_values)

        edges_data = []
        raw_scores = []

        safe_h_weight = self.h_weight_threshold if self.h_weight_threshold > 0 else 1.0
        safe_h_bridge = self.h_bridge_threshold if self.h_bridge_threshold > 0 else 1.0

        for u, v, edge_data in self.graph.edges(data=True):
            weight = float(edge_data.get("weight", 1.0))
            h_bridge = float(edge_data.get("h_bridge", 0.0))
            weight_ratio = weight / safe_h_weight
            bridge_ratio = h_bridge / safe_h_bridge
            score_raw = max(weight_ratio, bridge_ratio)
            in_backbone = bool(
                h_bridge >= self.h_bridge_threshold or weight >= self.h_weight_threshold
            )

            edges_data.append(
                {
                    "source": u,
                    "target": v,
                    "weight": weight,
                    "h_bridge": h_bridge,
                    "weight_ratio": weight_ratio,
                    "bridge_ratio": bridge_ratio,
                    "score_raw": score_raw,
                    "in_backbone": in_backbone,
                }
            )
            raw_scores.append(score_raw)

        score_min = min(raw_scores)
        score_max = max(raw_scores)

        final_edges = []
        self.score_measures = []
        self.alpha_measures = []
        for edge_info in edges_data:
            if score_max > score_min:
                score = (edge_info["score_raw"] - score_min) / (score_max - score_min)
            else:
                score = 1.0 if edge_info["in_backbone"] else 0.0
            alpha = 1.0 - score
            edge_info["score"] = float(score)
            edge_info["alpha"] = float(alpha)
            final_edges.append(edge_info)
            self.score_measures.append(float(score))
            self.alpha_measures.append(float(alpha))

        self.edges_df = pd.DataFrame(final_edges)
        if len(self.edges_df) > 0:
            alpha_scores = self.edges_df["alpha"].tolist()
            score_scores = self.edges_df["score"].tolist()
            self.edges_df["alpha_percentile"] = self.edges_df["alpha"].apply(
                lambda value: percentileofscore(alpha_scores, value) / 100.0
            )
            self.edges_df["score_percentile"] = self.edges_df["score"].apply(
                lambda value: percentileofscore(score_scores, value) / 100.0
            )
        else:
            self.edges_df["alpha_percentile"] = 0.0
            self.edges_df["score_percentile"] = 0.0

        self._create_nodes_dataframe()
        self._filter_applied = True
        return self.edges_df

    def filter_natural_backbone(self, min_degree: int = 1) -> nx.Graph:
        if not self._filter_applied:
            self.compute_filter()

        filtered_graph = self.graph.copy()
        edges_to_remove = [
            (row["source"], row["target"])
            for _, row in self.edges_df.iterrows()
            if not bool(row["in_backbone"])
        ]
        filtered_graph.remove_edges_from(edges_to_remove)

        self.nodesToKeep, nodes_to_remove = self._classify_nodes(
            filtered_graph,
            min_degree,
        )
        filtered_graph.remove_nodes_from(nodes_to_remove)
        return filtered_graph

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

        return {
            "num_nodes": len(self.graph.nodes()),
            "num_edges": len(self.graph.edges()),
            "h_weight_threshold": self.h_weight_threshold,
            "h_bridge_threshold": self.h_bridge_threshold,
            "score_mean": float(np.mean(self.score_measures)) if self.score_measures else 0.0,
            "score_median": float(np.median(self.score_measures)) if self.score_measures else 0.0,
            "score_std": float(np.std(self.score_measures)) if self.score_measures else 0.0,
            "alpha_mean": float(np.mean(self.alpha_measures)) if self.alpha_measures else 0.0,
            "alpha_median": float(np.median(self.alpha_measures)) if self.alpha_measures else 0.0,
            "alpha_std": float(np.std(self.alpha_measures)) if self.alpha_measures else 0.0,
        }

    def print_quantiles(self, num_quantiles: int = 10):
        if not self._filter_applied:
            self.compute_filter()

        bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
        quantiles = self.edges_df["alpha"].quantile(bins)

        print("\n" + "=" * 70)
        print("Quantis do H-Backbone")
        print("=" * 70)
        print(f"\n{'Percentil':<15} {'Alpha':<15} {'Interpretação'}")
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


def h_backbone(data):
    """
    Wrapper legado que preserva o comportamento anterior.

    Returns:
        Grafo anotado com `h_bridge` e `in_backbone`.
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

    h_filter = HBackboneFilter(graph)
    edges_df = h_filter.compute_filter()

    annotated_graph = graph.copy()
    for _, row in edges_df.iterrows():
        annotated_graph[row["source"]][row["target"]].update(
            {
                "h_bridge": row["h_bridge"],
                "in_backbone": bool(row["in_backbone"]),
                "score": row["score"],
                "alpha": row["alpha"],
            }
        )
    return annotated_graph

#!/usr/bin/env python
# encoding: utf-8

"""
Marginal Likelihood Filter para extração de backbone de redes complexas.

Baseado em um modelo nulo configuracional com teste binomial por aresta.
Esta implementação foi adaptada para a interface padrão do projeto.

Observação importante:
O método original foi formulado para pesos inteiros. Para permitir uso no
pipeline atual, esta versão pode converter pesos contínuos para inteiros por
reescala. Esse comportamento é uma adaptação pragmática e deve ser relatado
como tal ao interpretar resultados.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import binomtest, percentileofscore


logger = logging.getLogger(__name__)


class MarginalLikelihoodFilter:
    METHOD_NAME: str = "marg_likelihood"

    def __init__(
        self,
        graph: nx.Graph,
        *,
        directed: Optional[bool] = None,
        coerce_weights: bool = True,
        weight_scale: int = 1000,
    ):
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.directed = graph.is_directed() if directed is None else directed
        self.coerce_weights = coerce_weights
        self.weight_scale = weight_scale
        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.alpha_measures: list[float] = []
        self._filter_applied: bool = False
        self.nodesToKeep: list = []

    def _integerize_weight(self, weight: float) -> int:
        value = float(weight)
        if value < 0:
            raise ValueError("Marginal Likelihood Filter requer pesos nao-negativos.")

        if np.isclose(value, round(value)):
            return int(round(value))

        if not self.coerce_weights:
            raise ValueError(
                "Marginal Likelihood Filter requer pesos inteiros. "
                "Ative coerce_weights=True para usar a adaptacao por reescala."
            )

        scaled = int(round(value * self.weight_scale))
        if value > 0 and scaled < 1:
            scaled = 1
        return scaled

    def _compute_pvalue_undirected(self, w: int, ku: int, kv: int, q: int) -> float:
        if q <= 0:
            return 1.0
        p = (ku * kv) / (2.0 * q * q)
        p = min(max(p, 0.0), 1.0)
        return float(binomtest(k=w, n=q, p=p, alternative="greater").pvalue)

    def _compute_pvalue_directed(self, w_uv: int, ku_out: int, kv_in: int, q: int) -> float:
        if q <= 0:
            return 1.0
        p = (ku_out * kv_in) / (1.0 * q * q)
        p = min(max(p, 0.0), 1.0)
        return float(binomtest(k=w_uv, n=q, p=p, alternative="greater").pvalue)

    def compute_filter(self) -> pd.DataFrame:
        edges_data = []
        integer_weights = {}

        for u, v, edge_data in self.graph.edges(data=True):
            raw_weight = float(edge_data.get("weight", 1.0))
            integer_weight = self._integerize_weight(raw_weight)
            integer_weights[(u, v)] = integer_weight
            if not self.directed:
                integer_weights[(v, u)] = integer_weight

            edge_info = {
                "source": u,
                "target": v,
                "weight": raw_weight,
                "weight_integer": integer_weight,
            }
            if "pos" in self.graph.nodes[u]:
                edge_info["source_x"] = self.graph.nodes[u]["pos"][0]
                edge_info["source_y"] = self.graph.nodes[u]["pos"][1]
            if "pos" in self.graph.nodes[v]:
                edge_info["target_x"] = self.graph.nodes[v]["pos"][0]
                edge_info["target_y"] = self.graph.nodes[v]["pos"][1]
            edges_data.append(edge_info)

        table = pd.DataFrame(edges_data)
        if len(table) == 0:
            self.edges_df = table
            self.edges_df["p_value"] = []
            self.edges_df["alpha"] = []
            self.edges_df["alpha_percentile"] = []
            self._create_nodes_dataframe()
            self._filter_applied = True
            return self.edges_df

        if self.directed:
            out_strength = {
                node: 0 for node in self.graph.nodes()
            }
            in_strength = {
                node: 0 for node in self.graph.nodes()
            }
            total_weight = 0
            for u, v in self.graph.edges():
                weight_int = integer_weights[(u, v)]
                out_strength[u] += weight_int
                in_strength[v] += weight_int
                total_weight += weight_int

            p_values = []
            for _, row in table.iterrows():
                p_value = self._compute_pvalue_directed(
                    w_uv=int(row["weight_integer"]),
                    ku_out=out_strength[row["source"]],
                    kv_in=in_strength[row["target"]],
                    q=total_weight,
                )
                p_values.append(p_value)
        else:
            strength = {node: 0 for node in self.graph.nodes()}
            total_weight = 0
            for u, v in self.graph.edges():
                weight_int = integer_weights[(u, v)]
                strength[u] += weight_int
                strength[v] += weight_int
                total_weight += weight_int

            p_values = []
            for _, row in table.iterrows():
                p_value = self._compute_pvalue_undirected(
                    w=int(row["weight_integer"]),
                    ku=strength[row["source"]],
                    kv=strength[row["target"]],
                    q=total_weight,
                )
                p_values.append(p_value)

        table["p_value"] = p_values
        table["alpha"] = table["p_value"]

        alpha_scores = table["alpha"].tolist()
        table["alpha_percentile"] = table["alpha"].apply(
            lambda value: percentileofscore(alpha_scores, value) / 100.0
        )

        self.alpha_measures = table["alpha"].tolist()
        self.edges_df = table
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

        if not self.alpha_measures:
            return {
                "num_nodes": len(self.graph.nodes()),
                "num_edges": len(self.graph.edges()),
                "alpha_mean": 0.0,
                "alpha_median": 0.0,
                "alpha_std": 0.0,
                "alpha_min": 0.0,
                "alpha_max": 0.0,
            }

        return {
            "num_nodes": len(self.graph.nodes()),
            "num_edges": len(self.graph.edges()),
            "alpha_mean": float(np.mean(self.alpha_measures)),
            "alpha_median": float(np.median(self.alpha_measures)),
            "alpha_std": float(np.std(self.alpha_measures)),
            "alpha_min": float(np.min(self.alpha_measures)),
            "alpha_max": float(np.max(self.alpha_measures)),
        }

    def print_quantiles(self, num_quantiles: int = 10):
        if not self._filter_applied:
            self.compute_filter()

        bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
        quantiles = self.edges_df["alpha"].quantile(bins)

        print("\n" + "=" * 70)
        print("Quantis do Marginal Likelihood Filter")
        print("=" * 70)
        print(f"\n{'Percentil':<15} {'Alpha':<15} {'Interpretação'}")
        print("-" * 70)
        for percentile, value in quantiles.items():
            if value < 0.01:
                interpretation = "Muito significante"
            elif value < 0.05:
                interpretation = "Significante"
            else:
                interpretation = "Pouco significante"
            print(f"{percentile:>6.2%} {value:>15.6f} {interpretation}")
        print("=" * 70)


class MLF:
    """
    Wrapper de compatibilidade com a API antiga `fit_transform`.
    """

    def __init__(
        self,
        directed: bool = True,
        *,
        coerce_weights: bool = True,
        weight_scale: int = 1000,
    ):
        self.directed = directed
        self.coerce_weights = coerce_weights
        self.weight_scale = weight_scale

    def _to_networkx(self, graph):
        if isinstance(graph, nx.Graph):
            return graph.copy()
        if isinstance(graph, pd.DataFrame):
            return nx.from_pandas_edgelist(
                graph,
                source="source",
                target="target",
                edge_attr="weight",
                create_using=nx.DiGraph() if self.directed else nx.Graph(),
            )
        if isinstance(graph, list):
            nx_graph = nx.DiGraph() if self.directed else nx.Graph()
            nx_graph.add_weighted_edges_from(graph)
            return nx_graph
        if isinstance(graph, ig.Graph):
            nx_graph = nx.DiGraph() if self.directed else nx.Graph()
            weighted_edges = []
            for edge in graph.es:
                source = graph.vs[edge.source]["name"]
                target = graph.vs[edge.target]["name"]
                weight = edge["weight"] if "weight" in edge.attributes() else 1.0
                weighted_edges.append((source, target, weight))
            nx_graph.add_weighted_edges_from(weighted_edges)
            return nx_graph
        raise TypeError(
            "graph must be an instance of one of the following: "
            "igraph.Graph, list, pandas.DataFrame or networkx.Graph"
        )

    def fit_transform(self, graph):
        nx_graph = self._to_networkx(graph)
        ml_filter = MarginalLikelihoodFilter(
            nx_graph,
            directed=self.directed,
            coerce_weights=self.coerce_weights,
            weight_scale=self.weight_scale,
        )
        edges_df = ml_filter.compute_filter()

        if isinstance(graph, pd.DataFrame):
            return edges_df[["source", "target", "weight", "p_value"]].copy()
        if isinstance(graph, list):
            return list(
                edges_df[["source", "target", "weight", "p_value"]].itertuples(
                    index=False,
                    name=None,
                )
            )
        if isinstance(graph, ig.Graph):
            ig_graph = graph.copy()
            p_values = edges_df["p_value"].tolist()
            ig_graph.es["p_value"] = p_values
            return ig_graph

        annotated_graph = nx_graph.copy()
        for _, row in edges_df.iterrows():
            annotated_graph[row["source"]][row["target"]]["p_value"] = row["p_value"]
        return annotated_graph

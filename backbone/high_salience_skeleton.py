#!/usr/bin/env python
# encoding: utf-8

"""
High Salience Skeleton para extração de backbone de redes complexas.
Baseado em: Grady et al. (2012) - Robust classification of salient links in complex networks
https://www.nature.com/articles/ncomms1847
"""

from __future__ import annotations

import heapq
import itertools
from typing import Dict, Hashable, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


class HighSalienceSkeleton:
    """
    Classe para aplicar o High Salience Skeleton em redes complexas ponderadas.

    O método mede a fração de árvores de caminhos mínimos de origem única em que
    cada aresta aparece. Quanto maior a saliência, mais consenso existe entre os
    nós de que aquela aresta pertence ao esqueleto estrutural da rede.
    """

    def __init__(
        self,
        graph: nx.Graph,
        inverse_weight: bool = True,
        epsilon: float = 1e-12,
    ):
        """
        Inicializa o High Salience Skeleton com um grafo NetworkX.

        Args:
            graph: Grafo NetworkX ponderado
            inverse_weight: Se True, usa comprimento = 1 / peso
            epsilon: Tolerância numérica para empates de caminhos mínimos
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.inverse_weight = inverse_weight
        self.epsilon = epsilon
        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.salience_measures = []
        self.alpha_measures = []
        self._filter_applied = False
        self.nodesToKeep = []
        self._is_directed = self.graph.is_directed()

    def _edge_key(self, id0: Hashable, id1: Hashable):
        if self._is_directed:
            return (id0, id1)
        return frozenset((id0, id1))

    def _weight_to_length(self, weight: float) -> float:
        if weight < 0:
            raise ValueError(
                "High Salience Skeleton requer pesos nao-negativos para Dijkstra."
            )

        safe_weight = max(float(weight), self.epsilon)
        if self.inverse_weight:
            return 1.0 / safe_weight
        return safe_weight

    def _single_source_predecessors(
        self,
        source,
        edge_lengths: Dict,
    ) -> Dict[Hashable, list[Hashable]]:
        nodes = list(self.graph.nodes())
        distances = {node: np.inf for node in nodes}
        predecessors = {node: [] for node in nodes}
        queue = []
        push_index = itertools.count()

        distances[source] = 0.0
        heapq.heappush(queue, (0.0, next(push_index), source))

        while queue:
            current_distance, _, current = heapq.heappop(queue)
            if current_distance > distances[current] + self.epsilon:
                continue

            for neighbor in self.graph.neighbors(current):
                edge_key = self._edge_key(current, neighbor)
                candidate_distance = current_distance + edge_lengths[edge_key]

                if candidate_distance + self.epsilon < distances[neighbor]:
                    distances[neighbor] = candidate_distance
                    predecessors[neighbor] = [current]
                    heapq.heappush(
                        queue,
                        (candidate_distance, next(push_index), neighbor),
                    )
                elif abs(candidate_distance - distances[neighbor]) <= self.epsilon:
                    if current not in predecessors[neighbor]:
                        predecessors[neighbor].append(current)

        return predecessors

    def compute_filter(self) -> pd.DataFrame:
        """
        Calcula a saliência de todas as arestas da rede.

        Returns:
            DataFrame com informações detalhadas das arestas incluindo saliência
        """
        nodes = list(self.graph.nodes())
        num_sources = len(nodes)

        edge_lengths = {}
        salience_counts = {}

        for id0, id1, edge_data in self.graph.edges(data=True):
            weight = float(edge_data.get("weight", 1.0))
            edge_key = self._edge_key(id0, id1)
            edge_lengths[edge_key] = self._weight_to_length(weight)
            salience_counts[edge_key] = 0

        for source in nodes:
            predecessors = self._single_source_predecessors(source, edge_lengths)

            for node, preds in predecessors.items():
                for predecessor in preds:
                    salience_counts[self._edge_key(predecessor, node)] += 1

        edges_data = []
        self.salience_measures = []
        self.alpha_measures = []

        for id0, id1 in self.graph.edges():
            edge_data = self.graph[id0][id1]
            weight = float(edge_data.get("weight", 1.0))
            edge_key = self._edge_key(id0, id1)
            salience_count = salience_counts.get(edge_key, 0)
            salience = salience_count / num_sources if num_sources > 0 else 0.0
            alpha = 1.0 - salience

            edge_info = {
                "source": id0,
                "target": id1,
                "weight": weight,
                "length": edge_lengths[edge_key],
                "salience_count": salience_count,
                "salience": salience,
                "alpha": alpha,
            }

            if "pos" in self.graph.nodes[id0]:
                edge_info["source_x"] = self.graph.nodes[id0]["pos"][0]
                edge_info["source_y"] = self.graph.nodes[id0]["pos"][1]
            if "pos" in self.graph.nodes[id1]:
                edge_info["target_x"] = self.graph.nodes[id1]["pos"][0]
                edge_info["target_y"] = self.graph.nodes[id1]["pos"][1]

            edges_data.append(edge_info)
            self.salience_measures.append(salience)
            self.alpha_measures.append(alpha)

        self.edges_df = pd.DataFrame(edges_data)

        if len(self.edges_df) > 0:
            alpha_scores = self.edges_df["alpha"].tolist()
            salience_scores = self.edges_df["salience"].tolist()
            self.edges_df["alpha_percentile"] = self.edges_df["alpha"].apply(
                lambda value: percentileofscore(alpha_scores, value) / 100.0
            )
            self.edges_df["salience_percentile"] = self.edges_df["salience"].apply(
                lambda value: percentileofscore(salience_scores, value) / 100.0
            )
        else:
            self.edges_df["alpha_percentile"] = 0.0
            self.edges_df["salience_percentile"] = 0.0

        self._create_nodes_dataframe()
        self._filter_applied = True
        return self.edges_df

    def _create_nodes_dataframe(self) -> pd.DataFrame:
        """
        Cria DataFrame com informações dos nós.

        Returns:
            DataFrame com informações dos nós
        """
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
                if key not in ["pos", "label"]:
                    node_info[key] = value

            nodes_data.append(node_info)

        self.nodes_df = pd.DataFrame(nodes_data)
        return self.nodes_df

    def _edge_metric_lookup(self, column: str) -> Dict:
        if self.edges_df is None:
            raise ValueError("edges_df nao calculado. Execute compute_filter antes.")

        lookup = {}
        for _, row in self.edges_df.iterrows():
            lookup[self._edge_key(row["source"], row["target"])] = row[column]
        return lookup

    def filter_by_alpha(self, alpha: float, min_degree: int = 1) -> nx.Graph:
        """
        Aplica corte no grafo baseado no valor de alpha.

        Aqui alpha = 1 - salience, para manter compatibilidade com os outros
        filtros do projeto: quanto menor o alpha, mais significativa a aresta.

        Args:
            alpha: Threshold de significância. Arestas com alpha >= threshold
                são removidas
            min_degree: Grau mínimo para manter nó no grafo

        Returns:
            Grafo filtrado
        """
        if not self._filter_applied:
            self.compute_filter()

        filtered_graph = self.graph.copy()
        alpha_lookup = self._edge_metric_lookup("alpha")

        edges_to_remove = []
        for id0, id1 in filtered_graph.edges():
            edge_alpha = alpha_lookup.get(self._edge_key(id0, id1), 1.0)
            if edge_alpha >= alpha:
                edges_to_remove.append((id0, id1))

        filtered_graph.remove_edges_from(edges_to_remove)

        nodes_to_remove = [
            node for node in filtered_graph.nodes() if filtered_graph.degree(node) < min_degree
        ]
        self.nodesToKeep = [
            True if filtered_graph.degree(node) >= min_degree else False
            for node in filtered_graph.nodes()
        ]
        filtered_graph.remove_nodes_from(nodes_to_remove)

        return filtered_graph

    def filter_by_percentile(self, percentile: float, min_degree: int = 1) -> nx.Graph:
        """
        Aplica corte no grafo baseado no percentil de alpha.

        Args:
            percentile: Percentil para corte (0.0 a 1.0)
            min_degree: Grau mínimo para manter nó no grafo

        Returns:
            Grafo filtrado
        """
        if not self._filter_applied:
            self.compute_filter()

        filtered_graph = self.graph.copy()
        percentile_lookup = self._edge_metric_lookup("alpha_percentile")

        edges_to_remove = []
        for id0, id1 in filtered_graph.edges():
            edge_percentile = percentile_lookup.get(self._edge_key(id0, id1), 0.0)
            if edge_percentile < percentile:
                edges_to_remove.append((id0, id1))

        filtered_graph.remove_edges_from(edges_to_remove)

        nodes_to_remove = [
            node for node in filtered_graph.nodes() if filtered_graph.degree(node) < min_degree
        ]
        self.nodesToKeep = [
            True if filtered_graph.degree(node) >= min_degree else False
            for node in filtered_graph.nodes()
        ]
        filtered_graph.remove_nodes_from(nodes_to_remove)

        return filtered_graph

    def get_edges_dataframe(self) -> pd.DataFrame:
        """
        Retorna DataFrame com todas as informações das arestas.

        Returns:
            DataFrame com arestas e métricas
        """
        if not self._filter_applied:
            self.compute_filter()
        return self.edges_df

    def get_nodes_dataframe(self) -> pd.DataFrame:
        """
        Retorna DataFrame com todas as informações dos nós.

        Returns:
            DataFrame com nós e atributos
        """
        if self.nodes_df is None:
            self._create_nodes_dataframe()
        return self.nodes_df

    def get_summary_statistics(self) -> Dict:
        """
        Retorna estatísticas resumidas do filtro.

        Returns:
            Dicionário com estatísticas
        """
        if not self._filter_applied:
            self.compute_filter()

        if not self.salience_measures:
            return {
                "num_nodes": len(self.graph.nodes()),
                "num_edges": len(self.graph.edges()),
                "salience_mean": 0.0,
                "salience_median": 0.0,
                "salience_std": 0.0,
                "salience_min": 0.0,
                "salience_max": 0.0,
                "alpha_mean": 0.0,
                "alpha_median": 0.0,
                "alpha_std": 0.0,
                "alpha_min": 0.0,
                "alpha_max": 0.0,
            }

        return {
            "num_nodes": len(self.graph.nodes()),
            "num_edges": len(self.graph.edges()),
            "salience_mean": np.mean(self.salience_measures),
            "salience_median": np.median(self.salience_measures),
            "salience_std": np.std(self.salience_measures),
            "salience_min": np.min(self.salience_measures),
            "salience_max": np.max(self.salience_measures),
            "alpha_mean": np.mean(self.alpha_measures),
            "alpha_median": np.median(self.alpha_measures),
            "alpha_std": np.std(self.alpha_measures),
            "alpha_min": np.min(self.alpha_measures),
            "alpha_max": np.max(self.alpha_measures),
        }

    def print_quantiles(self, num_quantiles: int = 10):
        """
        Imprime os quantis de salience e alpha para ajudar na escolha do corte.

        Args:
            num_quantiles: Número de quantis a calcular
        """
        if not self._filter_applied:
            self.compute_filter()

        bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
        salience_quantiles = self.edges_df["salience"].quantile(bins)
        alpha_quantiles = self.edges_df["alpha"].quantile(bins)

        print("\n" + "=" * 70)
        print("Quantis do High Salience Skeleton")
        print("=" * 70)

        print(f"\n{'Percentil':<15} {'Salience':<15} {'Interpretação'}")
        print("-" * 70)
        for percentile, salience_value in salience_quantiles.items():
            if salience_value >= 0.8:
                interpretation = "Muito saliente"
            elif salience_value >= 0.5:
                interpretation = "Saliente"
            else:
                interpretation = "Pouco saliente"
            print(f"{percentile:>6.2%} {salience_value:>15.4f} {interpretation}")

        print(f"\n{'Percentil':<15} {'Alpha':<15} {'Interpretação'}")
        print("-" * 70)
        for percentile, alpha_value in alpha_quantiles.items():
            if alpha_value < 0.2:
                interpretation = "Muito saliente"
            elif alpha_value < 0.5:
                interpretation = "Saliente"
            else:
                interpretation = "Pouco saliente"
            print(f"{percentile:>6.2%} {alpha_value:>15.4f} {interpretation}")

        print("=" * 70)


if __name__ == "__main__":
    print("Criando grafo de exemplo (Les Misérables)...")
    G = nx.les_miserables_graph()

    print(f"\nGrafo original:")
    print(f"  Nós: {len(G.nodes())}")
    print(f"  Arestas: {len(G.edges())}")

    print("\nAplicando High Salience Skeleton...")
    hss = HighSalienceSkeleton(G)
    edges_df = hss.compute_filter()

    print("\nPrimeiras arestas ranqueadas por saliência:")
    print(edges_df.sort_values("salience", ascending=False).head())

    print("\nEstatísticas do filtro:")
    stats = hss.get_summary_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    hss.print_quantiles(num_quantiles=10)

    print("\nAplicando corte com alpha = 0.10...")
    filtered_graph = hss.filter_by_alpha(alpha=0.10, min_degree=1)

    print(f"\nGrafo filtrado (alpha=0.10):")
    print(f"  Nós: {len(filtered_graph.nodes())}")
    print(f"  Arestas: {len(filtered_graph.edges())}")

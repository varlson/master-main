#!/usr/bin/env python
# encoding: utf-8

"""
Doubly Stochastic Filter para extração de backbone de redes complexas.

Baseado em: Sinkhorn, R. (1964). A relationship between arbitrary positive
matrices and doubly stochastic matrices. Annals of Mathematical Statistics,
35(2), 876–879.

Aplicação em redes: Serrano et al. e NetBone (Zareie & Sakellariou, 2023).

O método normaliza iterativamente a matriz de adjacência ponderada até que
ela se torne duplamente estocástica (linhas e colunas somam 1). Os pesos
resultantes — chamados aqui de "scores" — refletem a importância relativa
de cada aresta após remoção do viés de grau e força dos nós. O backbone
é composto pelas arestas com maior score.

Corte natural (sem parâmetro):
    O backbone mínimo conexo é construído de forma gulosa: arestas são
    adicionadas em ordem decrescente de score até que o grafo resultante
    seja conexo e cubra todos os nós. Este é o único corte com motivação
    direta no método original.

Cortes adaptativos (por alpha e percentil):
    Adaptações desta implementação para uniformizar a interface com os
    demais métodos do projeto (DisparityFilter, HighSalienceSkeleton).
    O "alpha" é definido como (1 - score_normalizado), de modo que
    alpha baixo corresponde a alta importância — mesma semântica dos
    outros filtros. Esses cortes NÃO têm base direta no artigo original
    e devem ser identificados como adaptações metodológicas ao reportar
    resultados.

Notas sobre grafos direcionados:
    O método original foi proposto para grafos não-direcionados. Esta
    implementação suporta ambos os tipos, mas a normalização de Sinkhorn
    e a verificação de conectividade usam semânticas distintas para cada
    caso (conectividade forte vs. fraca). O uso com DiGraph não foi
    validado experimentalmente.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore


class DoublyStochasticFilter:
    """
    Aplica o Doubly Stochastic Filter em redes complexas ponderadas.

    O método transforma a matriz de adjacência ponderada em uma matriz
    duplamente estocástica por meio da normalização iterativa de Sinkhorn
    (1964). Os pesos normalizados resultantes (scores) refletem a
    importância estrutural de cada aresta independente do viés de grau
    e força dos nós.

    Backbone mínimo conexo:
        Além dos cortes paramétricos, o método fornece um backbone
        canônico construído de forma gulosa: arestas são adicionadas
        em ordem decrescente de score até que o grafo seja conexo.
        Esse backbone é acessível via `filter_connected_backbone()`.
    """

    # Tolerância de convergência para a normalização de Sinkhorn
    SINKHORN_TOL: float = 1e-12

    # Número máximo de iterações de Sinkhorn antes de emitir aviso
    SINKHORN_MAX_ITER: int = 1000

    # Regularização aditiva para evitar zeros na matriz (evita divisão por zero)
    SINKHORN_EPSILON: float = 1e-4

    # Identificador curto usado pelo pipeline de backbone.
    METHOD_NAME: str = "doub_stoch"

    def __init__(self, graph: nx.Graph):
        """
        Inicializa o DoublyStochasticFilter com um grafo NetworkX.

        Args:
            graph: Grafo NetworkX (direcionado ou não) com arestas ponderadas.
                   Espera-se que as arestas possuam o atributo 'weight'.
                   O método foi desenvolvido para grafos não-direcionados;
                   o suporte a DiGraph é parcial.
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.score_measures: list = []
        self.alpha_measures: list = []
        self._filter_applied: bool = False
        self._is_directed: bool = graph.is_directed()
        self.nodesToKeep: list = []
        self._sinkhorn_converged: bool = False
        self._sinkhorn_iterations: int = 0

    def _node_sort_key(self, node) -> tuple[str, str]:
        """
        Gera chave estável para ordenar endpoints de arestas não-direcionadas.

        A ordenação usa tipo + representação textual para evitar depender de
        comparações diretas entre objetos potencialmente heterogêneos.
        """
        return (type(node).__name__, str(node))

    def _edge_lookup_key(self, source, target):
        """
        Retorna chave canônica de aresta compatível com grafos dirigidos e não dirigidos.
        """
        if self._is_directed:
            return (source, target)

        if self._node_sort_key(source) <= self._node_sort_key(target):
            return (source, target)
        return (target, source)

    # ------------------------------------------------------------------
    # Normalização de Sinkhorn
    # ------------------------------------------------------------------

    def _build_weight_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Constrói a matriz de adjacência ponderada a partir do grafo.

        Aplica regularização aditiva (SINKHORN_EPSILON) para garantir
        que não haja zeros na matriz, pré-requisito da convergência de
        Sinkhorn para matrizes com suporte completo.

        Returns:
            Tupla (matrix_pivot, edgelist_original):
                - matrix_pivot: DataFrame N×N com pesos + regularização.
                - edgelist_original: DataFrame com (source, target, weight)
                  originais para merge posterior.
        """
        edgelist = nx.to_pandas_edgelist(self.graph)

        # Espelha as arestas para garantir simetria na tabela dinâmica
        # (necessário para grafos não-direcionados representados com
        #  source < target apenas)
        if not self._is_directed:
            mirror = edgelist.rename(columns={"source": "target", "target": "source"})
            edgelist_sym = pd.concat([edgelist, mirror], ignore_index=True)
        else:
            edgelist_sym = edgelist.copy()

        matrix = pd.pivot_table(
            edgelist_sym,
            values="weight",
            index="source",
            columns="target",
            aggfunc="sum",
            fill_value=0,
        ).astype(float)

        # Regularização: evita zeros que impediriam a convergência de Sinkhorn
        matrix += self.SINKHORN_EPSILON

        edgelist_original = edgelist[["source", "target", "weight"]].copy()
        if not self._is_directed:
            canonical_endpoints = edgelist_original.apply(
                lambda row: self._edge_lookup_key(row["source"], row["target"]),
                axis=1,
                result_type="expand",
            )
            canonical_endpoints.columns = ["source", "target"]
            edgelist_original[["source", "target"]] = canonical_endpoints
            edgelist_original = edgelist_original.drop_duplicates(
                subset=["source", "target"]
            ).reset_index(drop=True)

        return matrix, edgelist_original

    def _sinkhorn_normalize(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica a normalização iterativa de Sinkhorn até convergência.

        A cada iteração, normaliza linhas e depois colunas alternadamente.
        A convergência é atingida quando o desvio padrão das somas de linha
        cai abaixo de SINKHORN_TOL, indicando que a matriz é duplamente
        estocástica (todas as linhas e colunas somam aproximadamente 1).

        Se a convergência não for atingida em SINKHORN_MAX_ITER iterações,
        emite um RuntimeWarning e retorna a matriz no estado atual.

        Args:
            matrix: DataFrame N×N com pesos regularizados.

        Returns:
            DataFrame N×N duplamente estocástico (ou aproximação).
        """
        row_sums = matrix.sum(axis=1)
        self._sinkhorn_iterations = 0
        self._sinkhorn_converged = False

        while np.std(row_sums) > self.SINKHORN_TOL:
            matrix = matrix.div(row_sums, axis=0)
            col_sums = matrix.sum(axis=0)
            matrix = matrix.div(col_sums, axis=1)
            row_sums = matrix.sum(axis=1)
            self._sinkhorn_iterations += 1

            if self._sinkhorn_iterations > self.SINKHORN_MAX_ITER:
                warnings.warn(
                    "A matriz não convergiu para duplamente estocástica após "
                    f"{self.SINKHORN_MAX_ITER} iterações. "
                    "Veja Sinkhorn (1964), Sec. 3 para condições de convergência. "
                    "Os scores resultantes podem ser imprecisos.",
                    RuntimeWarning,
                )
                return matrix

        self._sinkhorn_converged = True
        return matrix

    def _matrix_to_edgelist(
        self,
        matrix: pd.DataFrame,
        edgelist_original: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Converte a matriz duplamente estocástica de volta para formato de arestas.

        Para grafos não-direcionados, mantém apenas pares com source <= target
        para evitar duplicatas. Descarta entradas com score zero.
        Realiza merge com os pesos originais para preservar o 'weight' bruto.

        Args:
            matrix: DataFrame N×N duplamente estocástico.
            edgelist_original: DataFrame com pesos originais.

        Returns:
            DataFrame de arestas com colunas (source, target, score, weight),
            ordenado por score decrescente.
        """
        melted = pd.melt(
            matrix.reset_index(),
            id_vars="source",
            var_name="target",
            value_name="score",
        )

        if not self._is_directed:
            canonical_endpoints = melted.apply(
                lambda row: self._edge_lookup_key(row["source"], row["target"]),
                axis=1,
                result_type="expand",
            )
            canonical_endpoints.columns = ["source", "target"]
            melted[["source", "target"]] = canonical_endpoints
            melted = melted.drop_duplicates(subset=["source", "target"])

        melted = melted[melted["score"] > 0].sort_values(
            by="score", ascending=False
        ).reset_index(drop=True)

        # Merge com pesos originais
        melted = melted.merge(
            edgelist_original,
            on=["source", "target"],
            how="left",
        )

        return melted

    # ------------------------------------------------------------------
    # Computação principal
    # ------------------------------------------------------------------

    def compute_filter(self) -> pd.DataFrame:
        """
        Executa a normalização de Sinkhorn e calcula os scores de todas
        as arestas.

        Para cada aresta, o score corresponde ao peso na matriz duplamente
        estocástica — quanto maior, mais estruturalmente relevante a aresta
        após remoção do viés de grau/força.

        O alpha é definido como (1 - score_normalizado), mantendo a
        convenção dos demais métodos do projeto: alpha baixo = alta
        significância = aresta a preservar.

        Returns:
            DataFrame com uma linha por aresta e as colunas:
            source, target, weight, score, score_normalizado,
            alpha, alpha_percentile, salience_percentile, in_backbone.
        """
        matrix, edgelist_original = self._build_weight_matrix()
        matrix = self._sinkhorn_normalize(matrix)
        table = self._matrix_to_edgelist(matrix, edgelist_original)

        # Backbone mínimo conexo (marcação gulosa)
        table = self._mark_connected_backbone(table)

        # Normalização do score para [0, 1] para calcular alpha comparável
        score_min = table["score"].min()
        score_max = table["score"].max()
        score_range = score_max - score_min

        if score_range > 0:
            table["score_normalized"] = (table["score"] - score_min) / score_range
        else:
            table["score_normalized"] = 1.0

        # Alpha: inversão do score normalizado (alpha baixo = score alto = preservar)
        table["alpha"] = 1.0 - table["score_normalized"]

        # Percentis
        alpha_scores = table["alpha"].tolist()
        score_scores = table["score"].tolist()

        table["alpha_percentile"] = table["alpha"].apply(
            lambda x: percentileofscore(alpha_scores, x) / 100.0
        )
        table["score_percentile"] = table["score"].apply(
            lambda x: percentileofscore(score_scores, x) / 100.0
        )

        # Adicionar coordenadas dos nós se disponíveis
        for prefix, col in [("source", "source"), ("target", "target")]:
            table[f"{col}_x"] = table[prefix].apply(
                lambda n: self.graph.nodes[n].get("pos", [None, None])[0]
                if n in self.graph.nodes else None
            )
            table[f"{col}_y"] = table[prefix].apply(
                lambda n: self.graph.nodes[n].get("pos", [None, None])[1]
                if n in self.graph.nodes else None
            )

        self.edges_df = table.reset_index(drop=True)
        self.score_measures = table["score"].tolist()
        self.alpha_measures = table["alpha"].tolist()

        self._create_nodes_dataframe()
        self._filter_applied = True
        return self.edges_df

    # ------------------------------------------------------------------
    # Backbone mínimo conexo (corte canônico do método)
    # ------------------------------------------------------------------

    def _mark_connected_backbone(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Marca as arestas que compõem o backbone mínimo conexo.

        Adiciona arestas em ordem decrescente de score até que o grafo
        resultante seja conexo e cubra todos os nós com score > 0.
        Este é o único corte com motivação direta no método original.

        Args:
            table: DataFrame de arestas ordenado por score decrescente.

        Returns:
            DataFrame com coluna 'in_backbone' (bool) adicionada.
        """
        table = table.copy()
        table["in_backbone"] = False

        target_nodes = set(table["source"]) | set(table["target"])
        n_target = len(target_nodes)
        total_edges = len(table)

        G = nx.DiGraph() if self._is_directed else nx.Graph()

        for i in range(total_edges):
            row = table.iloc[i]
            G.add_edge(row["source"], row["target"], weight=row["score"])
            table.at[table.index[i], "in_backbone"] = True

            # Critério de parada: grafo conexo cobrindo todos os nós
            if len(G) >= n_target:
                if self._is_directed:
                    connected = nx.is_weakly_connected(G)
                else:
                    connected = nx.is_connected(G)
                if connected:
                    break

        return table

    def filter_connected_backbone(self, min_degree: int = 1) -> nx.Graph:
        """
        Retorna o backbone mínimo conexo — corte canônico do método.

        Este é o backbone sem parâmetro livre: contém o menor conjunto
        de arestas com maior score que mantém o grafo conexo.

        Args:
            min_degree: Grau mínimo para preservação de nós.

        Returns:
            Grafo NetworkX com as arestas do backbone mínimo conexo.
        """
        if not self._filter_applied:
            self.compute_filter()

        backbone_edges = self.edges_df[self.edges_df["in_backbone"] == True]
        filtered_graph = nx.DiGraph() if self._is_directed else nx.Graph()

        for _, row in backbone_edges.iterrows():
            filtered_graph.add_edge(
                row["source"], row["target"],
                weight=row["weight"],
                score=row["score"],
                in_backbone=True,
            )

        # Transferir atributos dos nós do grafo original
        for node in filtered_graph.nodes():
            if node in self.graph.nodes:
                filtered_graph.nodes[node].update(self.graph.nodes[node])

        self.nodesToKeep, nodes_to_remove = self._classify_nodes(filtered_graph, min_degree)
        filtered_graph.remove_nodes_from(nodes_to_remove)

        return filtered_graph

    # ------------------------------------------------------------------
    # Cortes adaptativos (por alpha e percentil)
    # ------------------------------------------------------------------

    def filter_by_alpha(self, alpha: float, min_degree: int = 1) -> nx.Graph:
        """
        Filtra o grafo pelo alpha do Doubly Stochastic Filter.

        O alpha é definido como (1 - score_normalizado). Uma aresta é
        PRESERVADA se seu alpha for MENOR que o limiar α* — ou seja,
        se seu score normalizado for alto o suficiente.

            Preservar  ←→  alpha < α*  ←→  score_normalizado > (1 - α*)
            Remover    ←→  alpha >= α*

        ATENÇÃO: Este corte é uma adaptação desta implementação e NÃO
        tem base direta no artigo original. Para o corte canônico sem
        parâmetro, use `filter_connected_backbone()`.

        Args:
            alpha: Limiar α*. Arestas com alpha >= α* são removidas.
                   Valores típicos: 0.05, 0.10, 0.20.
            min_degree: Grau mínimo para preservação de nós.

        Returns:
            Novo grafo contendo as arestas do backbone.
        """
        if not self._filter_applied:
            self.compute_filter()

        edges_to_remove = [
            (row["source"], row["target"])
            for _, row in self.edges_df.iterrows()
            if row["alpha"] >= alpha
        ]

        return self._build_filtered_graph(edges_to_remove, min_degree)

    def filter_by_percentile(self, percentile: float, min_degree: int = 1) -> nx.Graph:
        """
        Filtra o grafo pelo percentil do alpha do Doubly Stochastic Filter.

        Como alpha baixo indica alta importância, preservar as arestas mais
        relevantes significa manter as de alpha_percentile BAIXO.

            Preservar  ←→  alpha_percentile <= percentile
            Remover    ←→  alpha_percentile > percentile

        Exemplos:
            percentile=0.30 → mantém as 30% de arestas com menor alpha
                              (maior score, mais relevantes).
            percentile=0.50 → mantém a metade mais relevante do grafo.

        ATENÇÃO: Este corte é uma adaptação desta implementação e NÃO
        tem base direta no artigo original. Para o corte canônico sem
        parâmetro, use `filter_connected_backbone()`.

        Args:
            percentile: Fração de arestas a preservar (0.0 a 1.0),
                        selecionando as de menor alpha (maior score).
            min_degree: Grau mínimo para preservação de nós.

        Returns:
            Novo grafo contendo as arestas do backbone.
        """
        if not self._filter_applied:
            self.compute_filter()

        edges_to_remove = [
            (row["source"], row["target"])
            for _, row in self.edges_df.iterrows()
            if row["alpha_percentile"] > percentile
        ]

        return self._build_filtered_graph(edges_to_remove, min_degree)

    # ------------------------------------------------------------------
    # Utilitários internos de filtragem
    # ------------------------------------------------------------------

    def _build_filtered_graph(
        self, edges_to_remove: list, min_degree: int
    ) -> nx.Graph:
        """
        Constrói grafo filtrado a partir de lista de arestas a remover.

        Args:
            edges_to_remove: Lista de tuplas (source, target).
            min_degree: Grau mínimo para preservação de nós.

        Returns:
            Grafo filtrado com atributos de nós preservados.
        """
        filtered_graph = self.graph.copy()
        filtered_graph.remove_edges_from(edges_to_remove)

        # Propagar score e in_backbone para as arestas remanescentes
        score_lookup = {
            self._edge_lookup_key(row["source"], row["target"]): {
                "score": row["score"],
                "alpha": row["alpha"],
                "in_backbone": row["in_backbone"],
            }
            for _, row in self.edges_df.iterrows()
        }
        for u, v in filtered_graph.edges():
            attrs = score_lookup.get(self._edge_lookup_key(u, v), {})
            filtered_graph[u][v].update(attrs)

        self.nodesToKeep, nodes_to_remove = self._classify_nodes(filtered_graph, min_degree)
        filtered_graph.remove_nodes_from(nodes_to_remove)

        return filtered_graph

    def _classify_nodes(
        self, filtered_graph: nx.Graph, min_degree: int
    ) -> Tuple[list, list]:
        """
        Classifica nós como a manter ou remover com base no grau mínimo.

        Itera sobre o grafo ORIGINAL para garantir alinhamento posicional
        com colunas do H5 (mesmo padrão do DisparityFilter e HSS).

        Args:
            filtered_graph: Grafo com arestas já filtradas.
            min_degree: Grau mínimo para preservação.

        Returns:
            Tupla (nodesToKeep, nodes_to_remove).
        """
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

    # ------------------------------------------------------------------
    # DataFrames
    # ------------------------------------------------------------------

    def _create_nodes_dataframe(self) -> pd.DataFrame:
        """
        Cria DataFrame com informações dos nós do grafo.

        Returns:
            DataFrame com atributos dos nós.
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
                if key not in ("pos", "label"):
                    node_info[key] = value
            nodes_data.append(node_info)

        self.nodes_df = pd.DataFrame(nodes_data)
        return self.nodes_df

    def get_edges_dataframe(self) -> pd.DataFrame:
        """
        Retorna DataFrame com todas as métricas das arestas.

        Returns:
            DataFrame com arestas, scores, alphas e percentis.
        """
        if not self._filter_applied:
            self.compute_filter()
        return self.edges_df

    def get_nodes_dataframe(self) -> pd.DataFrame:
        """
        Retorna DataFrame com informações dos nós.

        Returns:
            DataFrame com nós e atributos.
        """
        if self.nodes_df is None:
            self._create_nodes_dataframe()
        return self.nodes_df

    # ------------------------------------------------------------------
    # Estatísticas e diagnóstico
    # ------------------------------------------------------------------

    def get_summary_statistics(self) -> Dict:
        """
        Retorna estatísticas descritivas da distribuição de scores e alphas.

        Inclui informações sobre a convergência da normalização de Sinkhorn.

        Returns:
            Dicionário com contagens do grafo e estatísticas das métricas.
        """
        if not self._filter_applied:
            self.compute_filter()

        return {
            "num_nodes": len(self.graph.nodes()),
            "num_edges": len(self.graph.edges()),
            "sinkhorn_converged": self._sinkhorn_converged,
            "sinkhorn_iterations": self._sinkhorn_iterations,
            "score_mean": float(np.mean(self.score_measures)),
            "score_median": float(np.median(self.score_measures)),
            "score_std": float(np.std(self.score_measures)),
            "score_min": float(np.min(self.score_measures)),
            "score_max": float(np.max(self.score_measures)),
            "alpha_mean": float(np.mean(self.alpha_measures)),
            "alpha_median": float(np.median(self.alpha_measures)),
            "alpha_std": float(np.std(self.alpha_measures)),
            "alpha_min": float(np.min(self.alpha_measures)),
            "alpha_max": float(np.max(self.alpha_measures)),
        }

    def backbone_report(self, alpha: Optional[float] = None) -> Dict:
        """
        Gera relatório comparativo sobre o impacto do corte.

        Sempre reporta o backbone mínimo conexo. Se alpha for fornecido,
        também reporta o corte paramétrico correspondente.

        Args:
            alpha: Limiar opcional para corte adicional por alpha.

        Returns:
            Dicionário com contagens e taxas de redução.
        """
        if not self._filter_applied:
            self.compute_filter()

        n_original = len(self.graph.edges())
        n_backbone = int(self.edges_df["in_backbone"].sum())

        report = {
            "original_edges": n_original,
            "connected_backbone_edges": n_backbone,
            "connected_backbone_retention": round(n_backbone / n_original, 4) if n_original > 0 else 0.0,
            "sinkhorn_converged": self._sinkhorn_converged,
            "sinkhorn_iterations": self._sinkhorn_iterations,
        }

        if alpha is not None:
            n_alpha = int((self.edges_df["alpha"] < alpha).sum())
            report.update({
                "alpha_threshold": alpha,
                "alpha_backbone_edges": n_alpha,
                "alpha_backbone_retention": round(n_alpha / n_original, 4) if n_original > 0 else 0.0,
            })

        return report

    def print_quantiles(self, num_quantiles: int = 10):
        """
        Exibe os quantis da distribuição de scores e alphas para auxiliar
        na escolha do limiar de corte.

        Lembrete de interpretação:
            - Score alto → aresta relevante → preservar.
            - Alpha baixo → aresta relevante → preservar.

        Args:
            num_quantiles: Número de quantis a exibir.
        """
        if not self._filter_applied:
            self.compute_filter()

        bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
        score_quantiles = self.edges_df["score"].quantile(bins)
        alpha_quantiles = self.edges_df["alpha"].quantile(bins)

        print("\n" + "=" * 65)
        print("Doubly Stochastic Filter — Distribuição de Scores")
        print("Lembrete: score ALTO / alpha BAIXO = aresta RELEVANTE = PRESERVAR")
        print("=" * 65)

        print(f"\n{'Percentil':<15} {'Score':<15} {'Alpha':<12} {'Interpretação'}")
        print("-" * 65)

        for pct in bins:
            s_val = self.edges_df["score"].quantile(pct)
            a_val = self.edges_df["alpha"].quantile(pct)
            if a_val < 0.20:
                interp = "Muito relevante → preservar"
            elif a_val < 0.50:
                interp = "Relevante → preservar"
            elif a_val < 0.80:
                interp = "Limítrofe"
            else:
                interp = "Pouco relevante → remover"
            print(f"{pct:>6.2%}         {s_val:>10.6f}   {a_val:>8.4f}   {interp}")

        print("=" * 65)
        print(f"\nBackbone mínimo conexo: "
              f"{int(self.edges_df['in_backbone'].sum())} / {len(self.edges_df)} arestas")


# ------------------------------------------------------------------
# Exemplo de uso
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Criando grafo de exemplo (Les Misérables)...")
    # G = nx.les_miserables_graph()
    G = nx.read_graphml("../data/GraphML/metr-la.GraphML") 
    print(f"\nGrafo original: {len(G.nodes())} nós, {len(G.edges())} arestas")

    dsf = DoublyStochasticFilter(G)
    edges_df = dsf.compute_filter()

    print("\nEstatísticas do filtro:")
    for k, v in dsf.get_summary_statistics().items():
        print(f"  {k}: {v}")

    dsf.print_quantiles(num_quantiles=10)

    # Corte canônico: backbone mínimo conexo
    print("\n--- Backbone mínimo conexo (corte canônico) ---")
    backbone_conn = dsf.filter_connected_backbone(min_degree=1)
    print(f"Nós: {len(backbone_conn.nodes())}, Arestas: {len(backbone_conn.edges())}")

    # Relatório
    alpha = 0.8
    print(f"\nRelatório (alpha={alpha}):")
    for k, v in dsf.backbone_report(alpha=alpha).items():
        print(f"  {k}: {v}")

    # Corte por alpha (adaptativo)
    print(f"\n--- Corte por alpha={alpha} (adaptativo) ---")
    backbone_alpha = dsf.filter_by_alpha(alpha=alpha, min_degree=1)
    print(f"Nós: {len(backbone_alpha.nodes())}, Arestas: {len(backbone_alpha.edges())}")

    # Corte por percentil: mantém 30% mais relevantes
    print("\n--- Corte por percentil=0.30 (30% mais relevantes) ---")
    backbone_ptile = dsf.filter_by_percentile(percentile=0.30, min_degree=1)
    print(f"Nós: {len(backbone_ptile.nodes())}, Arestas: {len(backbone_ptile.edges())}")

    print("\nTop 10 arestas por score:")
    print(edges_df[["source", "target", "weight", "score", "alpha", "in_backbone"]].head(10))

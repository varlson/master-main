#!/usr/bin/env python
# encoding: utf-8

"""
Noise Corrected Filter para extração de backbone de redes complexas
Baseado em: Coscia & Neffke (2017) - Network Backboning with Noisy Data
"""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import binom, percentileofscore
from typing import Optional, Dict
import warnings


class NoiseCorrectedFilter:
    """
    Classe para aplicar o Noise Corrected Filter em redes complexas ponderadas.
    
    O filtro identifica arestas estatisticamente significantes corrigindo ruído
    através de modelagem bayesiana da distribuição de pesos.
    """
    
    def __init__(self, graph: nx.Graph, undirected: bool = False, 
                 return_self_loops: bool = False, use_p_value: bool = False):
        """
        Inicializa o Noise Corrected Filter com um grafo NetworkX.
        
        Args:
            graph: Grafo NetworkX (direcionado ou não) com arestas ponderadas
            undirected: Se True, considera grafo como não-direcionado
            return_self_loops: Se True, mantém self-loops no resultado
            use_p_value: Se True, usa p-value como score; caso contrário usa NC score
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.undirected = undirected
        self.return_self_loops = return_self_loops
        self.use_p_value = use_p_value
        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.score_measures = []
        self._filter_applied = False
        self.nodesToKeep = []
        
    def compute_filter(self) -> pd.DataFrame:
        """
        Calcula as métricas do noise corrected filter para todas as arestas.
        
        Returns:
            DataFrame com informações detalhadas das arestas incluindo scores
        """
        # Criar tabela inicial com arestas
        edges_data = []
        
        for id0, id1 in self.graph.edges():
            edge_data = self.graph[id0][id1]
            weight = edge_data.get('weight', 1.0)
            
            edge_info = {
                'src': id0,
                'trg': id1,
                'nij': weight,
                'weight': weight
            }
            
            # Adicionar coordenadas dos nós se disponíveis
            if 'pos' in self.graph.nodes[id0]:
                edge_info['src_x'] = self.graph.nodes[id0]['pos'][0]
                edge_info['src_y'] = self.graph.nodes[id0]['pos'][1]
            if 'pos' in self.graph.nodes[id1]:
                edge_info['trg_x'] = self.graph.nodes[id1]['pos'][0]
                edge_info['trg_y'] = self.graph.nodes[id1]['pos'][1]
            
            edges_data.append(edge_info)
        
        table = pd.DataFrame(edges_data)
        
        if len(table) == 0:
            self.edges_df = table
            self._filter_applied = True
            return self.edges_df
        
        # Calcular somas por src (ni.)
        src_sum = table.groupby(by="src").sum()[["nij"]]
        table = table.merge(src_sum, left_on="src", right_index=True, 
                           suffixes=("", "_src_sum"))
        
        # Calcular somas por trg (n.j)
        trg_sum = table.groupby(by="trg").sum()[["nij"]]
        table = table.merge(trg_sum, left_on="trg", right_index=True, 
                           suffixes=("", "_trg_sum"))
        
        table.rename(columns={"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, 
                    inplace=True)
        
        # Total de peso na rede (n..)
        table["n.."] = table["nij"].sum()
        
        # Probabilidade média a priori
        table["mean_prior_probability"] = (
            (table["ni."] * table["n.j"]) / table["n.."]
        ) * (1 / table["n.."])
        
        if self.use_p_value:
            # Calcular p-value usando distribuição binomial
            table["score"] = table.apply(
                lambda row: binom.cdf(row["nij"], row["n.."], 
                                     row["mean_prior_probability"]),
                axis=1
            )
            # Para p-value: valores menores = mais significante
            # Então alpha = p-value (como no disparity filter)
            table["alpha"] = table["score"]
        else:
            # Calcular NC score
            table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
            table["score"] = (
                (table["kappa"] * table["nij"]) - 1
            ) / ((table["kappa"] * table["nij"]) + 1)
            
            # Variância da probabilidade a priori
            table["var_prior_probability"] = (
                (1 / (table["n.."] ** 2)) * 
                (table["ni."] * table["n.j"] * 
                 (table["n.."] - table["ni."]) * 
                 (table["n.."] - table["n.j"])) / 
                ((table["n.."] ** 2) * (table["n.."] - 1))
            )
            
            # Parâmetros da distribuição Beta (prior)
            table["alpha_prior"] = (
                ((table["mean_prior_probability"] ** 2) / 
                 table["var_prior_probability"]) * 
                (1 - table["mean_prior_probability"])
            ) - table["mean_prior_probability"]
            
            table["beta_prior"] = (
                (table["mean_prior_probability"] / 
                 table["var_prior_probability"]) * 
                (1 - (table["mean_prior_probability"] ** 2))
            ) - (1 - table["mean_prior_probability"])
            
            # Parâmetros da distribuição Beta (posterior)
            table["alpha_post"] = table["alpha_prior"] + table["nij"]
            table["beta_post"] = (
                table["n.."] - table["nij"] + table["beta_prior"]
            )
            
            # Probabilidade esperada
            table["expected_pij"] = (
                table["alpha_post"] / 
                (table["alpha_post"] + table["beta_post"])
            )
            
            # Variância
            table["variance_nij"] = (
                table["expected_pij"] * 
                (1 - table["expected_pij"]) * 
                table["n.."]
            )
            
            table["d"] = (
                (1.0 / (table["ni."] * table["n.j"])) - 
                (table["n.."] * 
                 ((table["ni."] + table["n.j"]) / 
                  ((table["ni."] * table["n.j"]) ** 2)))
            )
            
            table["variance_cij"] = (
                table["variance_nij"] * 
                (((2 * (table["kappa"] + (table["nij"] * table["d"]))) / 
                  (((table["kappa"] * table["nij"]) + 1) ** 2)) ** 2)
            )
            
            table["sdev_cij"] = table["variance_cij"] ** 0.5
            
            # Para NC score: valores maiores = mais significante
            # Então alpha = 1 - normalized_score (para consistência com disparity)
            # Normalizando score para [0, 1]
            score_min = table["score"].min()
            score_max = table["score"].max()
            if score_max > score_min:
                table["score_normalized"] = (
                    (table["score"] - score_min) / (score_max - score_min)
                )
            else:
                table["score_normalized"] = 0.5
            
            # alpha: menor = mais significante (como disparity filter)
            table["alpha"] = 1 - table["score_normalized"]
        
        # Filtrar self-loops se necessário
        if not self.return_self_loops:
            table = table[table["src"] != table["trg"]]
        
        # Filtrar para grafo não-direcionado se necessário
        if self.undirected:
            table = table[table["src"] <= table["trg"]]
        
        # Coletar scores para estatísticas
        self.score_measures = table["score"].tolist()
        
        # Calcular percentis de alpha
        if len(table) > 0:
            table["alpha_percentile"] = table["alpha"].apply(
                lambda x: percentileofscore(table["alpha"].tolist(), x) / 100.0
            )
        else:
            table["alpha_percentile"] = 0.0
        
        # Renomear colunas para consistência
        table.rename(columns={"src": "source", "trg": "target"}, inplace=True)
        
        self.edges_df = table
        self._filter_applied = True
        
        # Criar DataFrame de nós
        self._create_nodes_dataframe()
        
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
                'node_id': node_id,
                'degree': self.graph.degree(node_id),
            }
            
            # Adicionar atributos do nó
            if 'pos' in node_data:
                node_info['x'] = node_data['pos'][0]
                node_info['y'] = node_data['pos'][1]
            
            if 'label' in node_data:
                node_info['label'] = node_data['label']
            
            # Adicionar outros atributos personalizados
            for key, value in node_data.items():
                if key not in ['pos', 'label']:
                    node_info[key] = value
            
            nodes_data.append(node_info)
        
        self.nodes_df = pd.DataFrame(nodes_data)
        return self.nodes_df
    
    def filter_by_alpha(self, alpha: float, min_degree: int = 1) -> nx.Graph:
        """
        Aplica corte no grafo baseado no valor de alpha.
        
        Para NC score: alpha representa 1 - score_normalizado
        Para p-value: alpha é o próprio p-value
        
        Args:
            alpha: Threshold de significância. Arestas com alpha >= threshold são removidas
            min_degree: Grau mínimo para manter nó no grafo
            
        Returns:
            Grafo filtrado
        """
        if not self._filter_applied:
            self.compute_filter()
        
        filtered_graph = self.graph.copy()
        
        # Remover arestas não significantes (alpha >= threshold)
        edges_to_remove = []
        for id0, id1 in filtered_graph.edges():
            edge_mask = (
                (self.edges_df['source'] == id0) & 
                (self.edges_df['target'] == id1)
            )
            edge_row = self.edges_df[edge_mask]
            
            if not edge_row.empty:
                edge_alpha = edge_row['alpha'].values[0]
                if edge_alpha >= alpha:
                    edges_to_remove.append((id0, id1))
        
        filtered_graph.remove_edges_from(edges_to_remove)
        
        # Remover nós com grau abaixo do mínimo
        nodes_to_remove = [
            node for node in filtered_graph.nodes() 
            if filtered_graph.degree(node) < min_degree
        ]
        self.nodesToKeep = [True if filtered_graph.degree(node) >= min_degree else False for node in filtered_graph.nodes()]
        
        filtered_graph.remove_nodes_from(nodes_to_remove)
        
        return filtered_graph
    
    def filter_by_percentile(self, percentile: float, min_degree: int = 1) -> nx.Graph:
        """
        Aplica corte no grafo baseado no percentil de alpha.
        
        Args:
            percentile: Percentil para corte (0.0 a 1.0). Ex: 0.5 mantém top 50% mais significantes
            min_degree: Grau mínimo para manter nó no grafo
            
        Returns:
            Grafo filtrado
        """
        if not self._filter_applied:
            self.compute_filter()
        
        filtered_graph = self.graph.copy()
        
        # Remover arestas abaixo do percentil
        edges_to_remove = []
        for id0, id1 in filtered_graph.edges():
            edge_mask = (
                (self.edges_df['source'] == id0) & 
                (self.edges_df['target'] == id1)
            )
            edge_row = self.edges_df[edge_mask]
            
            if not edge_row.empty:
                edge_percentile = edge_row['alpha_percentile'].values[0]
                if edge_percentile < percentile:
                    edges_to_remove.append((id0, id1))
        
        filtered_graph.remove_edges_from(edges_to_remove)
        
        # Remover nós com grau abaixo do mínimo
        nodes_to_remove = [
            node for node in filtered_graph.nodes() 
            if filtered_graph.degree(node) < min_degree
        ]
        
        self.nodesToKeep = [True if filtered_graph.degree(node) >= min_degree else False for node in filtered_graph.nodes()]
        
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
        
        stats = {
            'num_nodes': len(self.graph.nodes()),
            'num_edges': len(self.graph.edges()),
            'score_mean': np.mean(self.score_measures),
            'score_median': np.median(self.score_measures),
            'score_std': np.std(self.score_measures),
            'score_min': np.min(self.score_measures),
            'score_max': np.max(self.score_measures),
        }
        
        if 'alpha' in self.edges_df.columns:
            stats.update({
                'alpha_mean': self.edges_df['alpha'].mean(),
                'alpha_median': self.edges_df['alpha'].median(),
                'alpha_std': self.edges_df['alpha'].std(),
                'alpha_min': self.edges_df['alpha'].min(),
                'alpha_max': self.edges_df['alpha'].max(),
            })
        
        return stats
    
    def print_quantiles(self, num_quantiles: int = 10):
        """
        Imprime os quantis de score e alpha para ajudar na escolha do threshold.
        
        Args:
            num_quantiles: Número de quantis a calcular
        """
        if not self._filter_applied:
            self.compute_filter()
        
        bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
        
        print("\n" + "="*70)
        print(f"Quantis do Noise Corrected Filter ({'p-value' if self.use_p_value else 'NC score'})")
        print("="*70)
        
        # Quantis do score original
        score_quantiles = self.edges_df['score'].quantile(bins)
        print(f"\n{'Percentil':<15} {'Score':<15} {'Interpretação'}")
        print("-"*70)
        
        for percentile, score_val in score_quantiles.items():
            if self.use_p_value:
                interp = "Muito significante" if score_val < 0.01 else \
                         "Significante" if score_val < 0.05 else \
                         "Pouco significante"
            else:
                interp = "Pouco significante" if score_val < 0.3 else \
                         "Significante" if score_val < 0.7 else \
                         "Muito significante"
            print(f"{percentile:>6.2%} {score_val:>15.4f} {interp}")
        
        # Quantis do alpha (para consistência com disparity filter)
        if 'alpha' in self.edges_df.columns:
            alpha_quantiles = self.edges_df['alpha'].quantile(bins)
            print(f"\n{'Percentil':<15} {'Alpha':<15} {'Interpretação'}")
            print("-"*70)
            
            for percentile, alpha_val in alpha_quantiles.items():
                interp = "Muito significante" if alpha_val < 0.01 else \
                         "Significante" if alpha_val < 0.05 else \
                         "Pouco significante"
                print(f"{percentile:>6.2%} {alpha_val:>15.4f} {interp}")
        
        print("="*70)


# Exemplo de uso
if __name__ == "__main__":
    # Criar grafo de exemplo
    print("Criando grafo de exemplo (Les Misérables)...")
    G = nx.les_miserables_graph()
    
    print(f"\nGrafo original:")
    print(f"  Nós: {len(G.nodes())}")
    print(f"  Arestas: {len(G.edges())}")
    
    # Aplicar noise corrected filter (NC score)
    print("\n" + "="*70)
    print("Aplicando Noise Corrected Filter (NC Score)...")
    print("="*70)
    ncf = NoiseCorrectedFilter(G, undirected=True, use_p_value=False)
    
    # Computar métricas
    edges_df = ncf.compute_filter()
    
    # Mostrar estatísticas
    print("\nEstatísticas do filtro:")
    stats = ncf.get_summary_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Mostrar quantis
    ncf.print_quantiles(num_quantiles=10)
    
    # Aplicar filtro com alpha = 0.05
    print(f"\nAplicando corte com alpha = 0.05...")
    filtered_graph = ncf.filter_by_alpha(alpha=0.05, min_degree=0)
    
    print(f"\nGrafo filtrado (alpha=0.05):")
    print(f"  Nós: {len(filtered_graph.nodes())}")
    print(f"  Arestas: {len(filtered_graph.edges())}")
    print(f"  Redução de arestas: {(1 - len(filtered_graph.edges())/len(G.edges()))*100:.1f}%")
    
    # Aplicar filtro por percentil
    print(f"\nAplicando corte por percentil (top 30%)...")
    filtered_graph_ptile = ncf.filter_by_percentile(percentile=0.70, min_degree=0)
    
    print(f"\nGrafo filtrado (percentil 70%):")
    print(f"  Nós: {len(filtered_graph_ptile.nodes())}")
    print(f"  Arestas: {len(filtered_graph_ptile.edges())}")
    print(f"  Redução de arestas: {(1 - len(filtered_graph_ptile.edges())/len(G.edges()))*100:.1f}%")
    
    # Testar com p-value
    print("\n" + "="*70)
    print("Aplicando Noise Corrected Filter (P-Value)...")
    print("="*70)
    ncf_pval = NoiseCorrectedFilter(G, undirected=True, use_p_value=True)
    edges_df_pval = ncf_pval.compute_filter()
    ncf_pval.print_quantiles(num_quantiles=10)
    
    # Mostrar DataFrame de arestas (primeiras linhas)
    print("\nPrimeiras arestas do DataFrame (NC Score):")
    cols_to_show = ['source', 'target', 'weight', 'score', 'alpha', 'alpha_percentile']
    print(edges_df[[c for c in cols_to_show if c in edges_df.columns]].head(10))
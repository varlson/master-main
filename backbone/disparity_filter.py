#!/usr/bin/env python
# encoding: utf-8

"""
Disparity Filter para extração de backbone de redes complexas
Baseado em: Serrano et al. (2009) - Extracting the multiscale backbone of complex weighted networks
https://arxiv.org/pdf/0904.2389.pdf
"""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from typing import Optional, Tuple, Dict
import warnings
import h5py




class DisparityFilter:
    """
    Classe para aplicar o Disparity Filter em redes complexas ponderadas.
    
    O filtro identifica arestas estatisticamente significantes baseado na
    distribuição de pesos das conexões de cada nó.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Inicializa o Disparity Filter com um grafo NetworkX.
        
        Args:
            graph: Grafo NetworkX (direcionado ou não) com arestas ponderadas
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.alpha_measures = []
        self._filter_applied = False
        self.nodeMapping: Dict = {}
        self.nodeIDs = list(nx.get_edge_attributes(self.graph, 'nodeId').values())
        self.nodesToKeep = []
        
    def _disparity_integral(self, x: float, k: int) -> float:
        """
        Calcula a integral definida para o PDF do disparity filter.
        
        Args:
            x: Peso normalizado
            k: Grau do nó
            
        Returns:
            Valor da integral
        """
        if x == 1.0 or k == 1:
            return 0.0
        return ((1.0 - x) ** k) / ((k - 1.0) * (x - 1.0))
    
    def _get_disparity_significance(self, norm_weight: float, degree: int) -> float:
        """
        Calcula o p-value (significância) para o disparity filter.
        
        Args:
            norm_weight: Peso normalizado da aresta
            degree: Grau do nó
            
        Returns:
            Valor de alpha (p-value)
        """
        if degree <= 1:
            return 0.0
            
        # Ajuste para evitar norm_weight = 1.0
        if norm_weight >= 0.9999:
            norm_weight = 0.9999
            
        try:
            integral_x = self._disparity_integral(norm_weight, degree)
            integral_0 = self._disparity_integral(0.0, degree)
            alpha = 1.0 - ((degree - 1.0) * (integral_x - integral_0))
            return max(0.0, min(1.0, alpha))  # Garante que alpha está entre 0 e 1
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    def compute_filter(self) -> pd.DataFrame:
        """
        Calcula as métricas do disparity filter para todas as arestas.
        
        Returns:
            DataFrame com informações detalhadas das arestas incluindo p-values
        """
        edges_data = []
        self.alpha_measures = []
        
        # Primeira passagem: calcular strength de cada nó
        node_strength = {}
        for node_id in self.graph.nodes():
            strength = 0.0
            for id0, id1 in self.graph.edges(node_id):
                edge_data = self.graph[id0][id1]
                weight = edge_data.get('weight', 1.0)
                strength += weight
            node_strength[node_id] = strength
        
        # Segunda passagem: calcular métricas das arestas
        for id0, id1 in self.graph.edges():
            edge_data = self.graph[id0][id1]
            weight = edge_data.get('weight', 1.0)
            
            # Informações básicas da aresta
            edge_info = {
                'source': id0,
                'target': id1,
                'weight': weight
            }
            
            # Adicionar coordenadas dos nós se disponíveis
            if 'pos' in self.graph.nodes[id0]:
                edge_info['source_x'] = self.graph.nodes[id0]['pos'][0]
                edge_info['source_y'] = self.graph.nodes[id0]['pos'][1]
            if 'pos' in self.graph.nodes[id1]:
                edge_info['target_x'] = self.graph.nodes[id1]['pos'][0]
                edge_info['target_y'] = self.graph.nodes[id1]['pos'][1]
            
            # Calcular alpha para ambas as direções (source e target)
            degree_source = self.graph.degree(id0)
            degree_target = self.graph.degree(id1)
            
            strength_source = node_strength[id0]
            strength_target = node_strength[id1]
            
            # Peso normalizado e alpha do source
            norm_weight_source = weight / strength_source if strength_source > 0 else 0
            alpha_source = self._get_disparity_significance(norm_weight_source, degree_source)
            
            # Peso normalizado e alpha do target
            norm_weight_target = weight / strength_target if strength_target > 0 else 0
            alpha_target = self._get_disparity_significance(norm_weight_target, degree_target)
            
            # Usar o maior alpha (mais conservador)
            alpha = max(alpha_source, alpha_target)
            
            edge_info.update({
                'degree_source': degree_source,
                'degree_target': degree_target,
                'strength_source': strength_source,
                'strength_target': strength_target,
                'norm_weight_source': norm_weight_source,
                'norm_weight_target': norm_weight_target,
                'alpha_source': alpha_source,
                'alpha_target': alpha_target,
                'alpha': alpha,  # p-value para uso nos cortes
            })
            
            edges_data.append(edge_info)
            self.alpha_measures.append(alpha)
        
        # Criar DataFrame
        self.edges_df = pd.DataFrame(edges_data)
        
        # Calcular percentis
        if len(self.alpha_measures) > 0:
            self.edges_df['alpha_percentile'] = self.edges_df['alpha'].apply(
                lambda x: percentileofscore(self.alpha_measures, x) / 100.0
            )
        else:
            self.edges_df['alpha_percentile'] = 0.0
        
        # Adicionar informações dos nós
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
        Aplica corte no grafo baseado no valor de alpha (p-value).
        
        Args:
            alpha: Threshold de significância (p-value). Arestas com alpha >= threshold são removidas
            min_degree: Grau mínimo para manter nó no grafo
            
        Returns:
            Grafo filtrado
        """
        if not self._filter_applied:
            self.compute_filter()
        
        filtered_graph = self.graph.copy()
        
        # Remover arestas não significantes
        edges_to_remove = []
        for id0, id1 in filtered_graph.edges():
            edge_mask = (
                (self.edges_df['source'] == id0) & 
                (self.edges_df['target'] == id1)
            )
            edge_row = self.edges_df[edge_mask]
            
            if not edge_row.empty:
                edge_alpha = edge_row['alpha'].values[0]
                if edge_alpha >= alpha:  # Remove arestas não significantes
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
    
    def rebuildH5Data(self, filepath: str, key: str = "df") -> np.ndarray:
        if not self.nodesToKeep:
            raise ValueError(
                "nodesToKeep vazio. Execute filter_by_alpha/filter_by_percentile antes."
            )

        with h5py.File(filepath, "r") as file_obj:
            if key not in file_obj:
                available_keys = list(file_obj.keys())
                raise KeyError(
                    f"Chave '{key}' nao encontrada no H5. Disponiveis: {available_keys}"
                )
            data = np.array(file_obj[key]["block0_values"])

        if len(self.nodesToKeep) != data.shape[1]:
            raise ValueError(
                "nodesToKeep e quantidade de colunas do H5 nao batem: "
                f"{len(self.nodesToKeep)} != {data.shape[1]}"
            )

        keep_mask = np.array(self.nodesToKeep, dtype=bool)
        return data[:, keep_mask]
    
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
        
        return {
            'num_nodes': len(self.graph.nodes()),
            'num_edges': len(self.graph.edges()),
            'alpha_mean': np.mean(self.alpha_measures),
            'alpha_median': np.median(self.alpha_measures),
            'alpha_std': np.std(self.alpha_measures),
            'alpha_min': np.min(self.alpha_measures),
            'alpha_max': np.max(self.alpha_measures),
        }
    
    def print_quantiles(self, num_quantiles: int = 10):
        """
        Imprime os quantis de alpha para ajudar na escolha do threshold.
        
        Args:
            num_quantiles: Número de quantis a calcular
        """
        if not self._filter_applied:
            self.compute_filter()
        
        bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
        quantiles = self.edges_df['alpha'].quantile(bins)
        
        print("\n" + "="*50)
        print("Quantis de Alpha (p-value)")
        print("="*50)
        print(f"{'Percentil':<15} {'Alpha':<15} {'Interpretação'}")
        print("-"*50)
        
        for percentile, alpha_val in quantiles.items():
            interp = "Muito significante" if alpha_val < 0.01 else \
                     "Significante" if alpha_val < 0.05 else \
                     "Pouco significante"
            print(f"{percentile:>6.2%} {alpha_val:>15.4f} {interp}")
        
        print("="*50)


# Exemplo de uso
if __name__ == "__main__":
    # Criar grafo de exemplo
    print("Criando grafo de exemplo (Les Misérables)...")
    G = nx.read_graphml('aerial.GraphML')
    # G = nx.les_miserables_graph()
    
    print(f"\nGrafo original:")
    print(f"  Nós: {len(G.nodes())}")
    print(f"  Arestas: {len(G.edges())}")
    
    # Aplicar disparity filter
    print("\nAplicando Disparity Filter...")
    df = DisparityFilter(G)
    
    # Computar métricas
    edges_df = df.compute_filter()
    
    # Mostrar estatísticas
    print("\nEstatísticas do filtro:")
    stats = df.get_summary_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Mostrar quantis
    df.print_quantiles(num_quantiles=10)
    
    # Aplicar filtro com alpha = 0.05 (significância de 5%)
    print(f"\nAplicando corte com alpha = 0.05...")
    filtered_graph = df.filter_by_alpha(alpha=0.05, min_degree=1)
    
    print(f"\nGrafo filtrado (alpha=0.05):")
    print(f"  Nós: {len(filtered_graph.nodes())}")
    print(f"  Arestas: {len(filtered_graph.edges())}")
    print(f"  Redução de arestas: {(1 - len(filtered_graph.edges())/len(G.edges()))*100:.1f}%")
    
    # Aplicar filtro por percentil
    print(f"\nAplicando corte por percentil (top 30%)...")
    filtered_graph_ptile = df.filter_by_percentile(percentile=0.70, min_degree=0)
    
    print(f"\nGrafo filtrado (percentil 70%):")
    print(f"  Nós: {len(filtered_graph_ptile.nodes())}")
    print(f"  Arestas: {len(filtered_graph_ptile.edges())}")
    print(f"  Redução de arestas: {(1 - len(filtered_graph_ptile.edges())/len(G.edges()))*100:.1f}%")
    
    # Mostrar DataFrame de arestas (primeiras linhas)
    print("\nPrimeiras arestas do DataFrame:")
    print(edges_df[['source', 'target', 'weight', 'alpha', 'alpha_percentile']].head(10))

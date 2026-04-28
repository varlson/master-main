# #!/usr/bin/env python
# # encoding: utf-8

# """
# Disparity Filter para extração de backbone de redes complexas
# Baseado em: Serrano et al. (2009) - Extracting the multiscale backbone of complex weighted networks
# https://arxiv.org/pdf/0904.2389.pdf
# """

# import networkx as nx
# import numpy as np
# import pandas as pd
# from scipy.stats import percentileofscore
# from typing import Optional, Tuple, Dict
# import warnings
# import h5py




# class DisparityFilter:
#     """
#     Classe para aplicar o Disparity Filter em redes complexas ponderadas.
    
#     O filtro identifica arestas estatisticamente significantes baseado na
#     distribuição de pesos das conexões de cada nó.
#     """
    
#     def __init__(self, graph: nx.Graph):
#         """
#         Inicializa o Disparity Filter com um grafo NetworkX.
        
#         Args:
#             graph: Grafo NetworkX (direcionado ou não) com arestas ponderadas
#         """
#         self.original_graph = graph.copy()
#         self.graph = graph.copy()
#         self.edges_df: Optional[pd.DataFrame] = None
#         self.nodes_df: Optional[pd.DataFrame] = None
#         self.alpha_measures = []
#         self._filter_applied = False
#         self.nodeMapping: Dict = {}
#         self.nodeIDs = list(nx.get_edge_attributes(self.graph, 'nodeId').values())
#         self.nodesToKeep = []
        
#     def _disparity_integral(self, x: float, k: int) -> float:
#         """
#         Calcula a integral definida para o PDF do disparity filter.
        
#         Args:
#             x: Peso normalizado
#             k: Grau do nó
            
#         Returns:
#             Valor da integral
#         """
#         if x == 1.0 or k == 1:
#             return 0.0
#         return ((1.0 - x) ** k) / ((k - 1.0) * (x - 1.0))
    
#     def _get_disparity_significance(self, norm_weight: float, degree: int) -> float:
#         """
#         Calcula o p-value (significância) para o disparity filter.
        
#         Args:
#             norm_weight: Peso normalizado da aresta
#             degree: Grau do nó
            
#         Returns:
#             Valor de alpha (p-value)
#         """
#         if degree <= 1:
#             return 0.0
            
#         # Ajuste para evitar norm_weight = 1.0
#         if norm_weight >= 0.9999:
#             norm_weight = 0.9999
            
#         try:
#             integral_x = self._disparity_integral(norm_weight, degree)
#             integral_0 = self._disparity_integral(0.0, degree)
#             alpha = 1.0 - ((degree - 1.0) * (integral_x - integral_0))
#             return max(0.0, min(1.0, alpha))  # Garante que alpha está entre 0 e 1
#         except (ZeroDivisionError, ValueError):
#             return 0.0
    
#     def compute_filter(self) -> pd.DataFrame:
#         """
#         Calcula as métricas do disparity filter para todas as arestas.
        
#         Returns:
#             DataFrame com informações detalhadas das arestas incluindo p-values
#         """
#         edges_data = []
#         self.alpha_measures = []
        
#         # Primeira passagem: calcular strength de cada nó
#         node_strength = {}
#         for node_id in self.graph.nodes():
#             strength = 0.0
#             for id0, id1 in self.graph.edges(node_id):
#                 edge_data = self.graph[id0][id1]
#                 weight = edge_data.get('weight', 1.0)
#                 strength += weight
#             node_strength[node_id] = strength
        
#         # Segunda passagem: calcular métricas das arestas
#         for id0, id1 in self.graph.edges():
#             edge_data = self.graph[id0][id1]
#             weight = edge_data.get('weight', 1.0)
            
#             # Informações básicas da aresta
#             edge_info = {
#                 'source': id0,
#                 'target': id1,
#                 'weight': weight
#             }
            
#             # Adicionar coordenadas dos nós se disponíveis
#             if 'pos' in self.graph.nodes[id0]:
#                 edge_info['source_x'] = self.graph.nodes[id0]['pos'][0]
#                 edge_info['source_y'] = self.graph.nodes[id0]['pos'][1]
#             if 'pos' in self.graph.nodes[id1]:
#                 edge_info['target_x'] = self.graph.nodes[id1]['pos'][0]
#                 edge_info['target_y'] = self.graph.nodes[id1]['pos'][1]
            
#             # Calcular alpha para ambas as direções (source e target)
#             degree_source = self.graph.degree(id0)
#             degree_target = self.graph.degree(id1)
            
#             strength_source = node_strength[id0]
#             strength_target = node_strength[id1]
            
#             # Peso normalizado e alpha do source
#             norm_weight_source = weight / strength_source if strength_source > 0 else 0
#             alpha_source = self._get_disparity_significance(norm_weight_source, degree_source)
            
#             # Peso normalizado e alpha do target
#             norm_weight_target = weight / strength_target if strength_target > 0 else 0
#             alpha_target = self._get_disparity_significance(norm_weight_target, degree_target)
            
#             # Usar o maior alpha (mais conservador)
#             alpha = max(alpha_source, alpha_target)
            
#             edge_info.update({
#                 'degree_source': degree_source,
#                 'degree_target': degree_target,
#                 'strength_source': strength_source,
#                 'strength_target': strength_target,
#                 'norm_weight_source': norm_weight_source,
#                 'norm_weight_target': norm_weight_target,
#                 'alpha_source': alpha_source,
#                 'alpha_target': alpha_target,
#                 'alpha': alpha,  # p-value para uso nos cortes
#             })
            
#             edges_data.append(edge_info)
#             self.alpha_measures.append(alpha)
        
#         # Criar DataFrame
#         self.edges_df = pd.DataFrame(edges_data)
        
#         # Calcular percentis
#         if len(self.alpha_measures) > 0:
#             self.edges_df['alpha_percentile'] = self.edges_df['alpha'].apply(
#                 lambda x: percentileofscore(self.alpha_measures, x) / 100.0
#             )
#         else:
#             self.edges_df['alpha_percentile'] = 0.0
        
#         # Adicionar informações dos nós
#         self._create_nodes_dataframe()
        
#         self._filter_applied = True
#         return self.edges_df
    
#     def _create_nodes_dataframe(self) -> pd.DataFrame:
#         """
#         Cria DataFrame com informações dos nós.
        
#         Returns:
#             DataFrame com informações dos nós
#         """
#         nodes_data = []
        
#         for node_id in self.graph.nodes():
#             node_data = self.graph.nodes[node_id]
#             node_info = {
#                 'node_id': node_id,
#                 'degree': self.graph.degree(node_id),
#             }
            
#             # Adicionar atributos do nó
#             if 'pos' in node_data:
#                 node_info['x'] = node_data['pos'][0]
#                 node_info['y'] = node_data['pos'][1]
            
#             if 'label' in node_data:
#                 node_info['label'] = node_data['label']
            
#             # Adicionar outros atributos personalizados
#             for key, value in node_data.items():
#                 if key not in ['pos', 'label']:
#                     node_info[key] = value
            
#             nodes_data.append(node_info)
        
#         self.nodes_df = pd.DataFrame(nodes_data)
#         return self.nodes_df
    
#     def filter_by_alpha(self, alpha: float, min_degree: int = 1) -> nx.Graph:
#         """
#         Aplica corte no grafo baseado no valor de alpha (p-value).
        
#         Args:
#             alpha: Threshold de significância (p-value). Arestas com alpha >= threshold são removidas
#             min_degree: Grau mínimo para manter nó no grafo
            
#         Returns:
#             Grafo filtrado
#         """
#         if not self._filter_applied:
#             self.compute_filter()
        
#         filtered_graph = self.graph.copy()
        
#         # Remover arestas não significantes
#         edges_to_remove = []
#         for id0, id1 in filtered_graph.edges():
#             edge_mask = (
#                 (self.edges_df['source'] == id0) & 
#                 (self.edges_df['target'] == id1)
#             )
#             edge_row = self.edges_df[edge_mask]
            
#             if not edge_row.empty:
#                 edge_alpha = edge_row['alpha'].values[0]
#                 if edge_alpha >= alpha:  # Remove arestas não significantes
#                     edges_to_remove.append((id0, id1))
        
#         filtered_graph.remove_edges_from(edges_to_remove)
        
#         # Remover nós com grau abaixo do mínimo
#         nodes_to_remove = [
#             node for node in filtered_graph.nodes() 
#             if filtered_graph.degree(node) < min_degree
#         ]
#         self.nodesToKeep = [True if filtered_graph.degree(node) >= min_degree else False for node in filtered_graph.nodes()]
#         filtered_graph.remove_nodes_from(nodes_to_remove)
        
#         return filtered_graph
    
#     def filter_by_percentile(self, percentile: float, min_degree: int = 1) -> nx.Graph:
#         """
#         Aplica corte no grafo baseado no percentil de alpha.
        
#         Args:
#             percentile: Percentil para corte (0.0 a 1.0). Ex: 0.5 mantém top 50% mais significantes
#             min_degree: Grau mínimo para manter nó no grafo
            
#         Returns:
#             Grafo filtrado
#         """
#         if not self._filter_applied:
#             self.compute_filter()
        
#         filtered_graph = self.graph.copy()
        
#         # Remover arestas abaixo do percentil
#         edges_to_remove = []
#         for id0, id1 in filtered_graph.edges():
#             edge_mask = (
#                 (self.edges_df['source'] == id0) & 
#                 (self.edges_df['target'] == id1)
#             )
#             edge_row = self.edges_df[edge_mask]
            
#             if not edge_row.empty:
#                 edge_percentile = edge_row['alpha_percentile'].values[0]
#                 if edge_percentile < percentile:
#                     edges_to_remove.append((id0, id1))
        
#         filtered_graph.remove_edges_from(edges_to_remove)
        
#         # Remover nós com grau abaixo do mínimo
#         nodes_to_remove = [
#             node for node in filtered_graph.nodes() 
#             if filtered_graph.degree(node) < min_degree
#         ]
#         self.nodesToKeep = [True if filtered_graph.degree(node) >= min_degree else False for node in filtered_graph.nodes()]
#         filtered_graph.remove_nodes_from(nodes_to_remove)
        
#         return filtered_graph
    
#     def rebuildH5Data(self, filepath: str, key: str = "df") -> np.ndarray:
#         if not self.nodesToKeep:
#             raise ValueError(
#                 "nodesToKeep vazio. Execute filter_by_alpha/filter_by_percentile antes."
#             )

#         with h5py.File(filepath, "r") as file_obj:
#             if key not in file_obj:
#                 available_keys = list(file_obj.keys())
#                 raise KeyError(
#                     f"Chave '{key}' nao encontrada no H5. Disponiveis: {available_keys}"
#                 )
#             data = np.array(file_obj[key]["block0_values"])

#         if len(self.nodesToKeep) != data.shape[1]:
#             raise ValueError(
#                 "nodesToKeep e quantidade de colunas do H5 nao batem: "
#                 f"{len(self.nodesToKeep)} != {data.shape[1]}"
#             )

#         keep_mask = np.array(self.nodesToKeep, dtype=bool)
#         return data[:, keep_mask]
    
#     def get_edges_dataframe(self) -> pd.DataFrame:
#         """
#         Retorna DataFrame com todas as informações das arestas.
        
#         Returns:
#             DataFrame com arestas e métricas
#         """
#         if not self._filter_applied:
#             self.compute_filter()
#         return self.edges_df
    
#     def get_nodes_dataframe(self) -> pd.DataFrame:
#         """
#         Retorna DataFrame com todas as informações dos nós.
        
#         Returns:
#             DataFrame com nós e atributos
#         """
#         if self.nodes_df is None:
#             self._create_nodes_dataframe()
#         return self.nodes_df
    
#     def get_summary_statistics(self) -> Dict:
#         """
#         Retorna estatísticas resumidas do filtro.
        
#         Returns:
#             Dicionário com estatísticas
#         """
#         if not self._filter_applied:
#             self.compute_filter()
        
#         return {
#             'num_nodes': len(self.graph.nodes()),
#             'num_edges': len(self.graph.edges()),
#             'alpha_mean': np.mean(self.alpha_measures),
#             'alpha_median': np.median(self.alpha_measures),
#             'alpha_std': np.std(self.alpha_measures),
#             'alpha_min': np.min(self.alpha_measures),
#             'alpha_max': np.max(self.alpha_measures),
#         }
    
#     def print_quantiles(self, num_quantiles: int = 10):
#         """
#         Imprime os quantis de alpha para ajudar na escolha do threshold.
        
#         Args:
#             num_quantiles: Número de quantis a calcular
#         """
#         if not self._filter_applied:
#             self.compute_filter()
        
#         bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
#         quantiles = self.edges_df['alpha'].quantile(bins)
        
#         print("\n" + "="*50)
#         print("Quantis de Alpha (p-value)")
#         print("="*50)
#         print(f"{'Percentil':<15} {'Alpha':<15} {'Interpretação'}")
#         print("-"*50)
        
#         for percentile, alpha_val in quantiles.items():
#             interp = "Muito significante" if alpha_val < 0.01 else \
#                      "Significante" if alpha_val < 0.05 else \
#                      "Pouco significante"
#             print(f"{percentile:>6.2%} {alpha_val:>15.4f} {interp}")
        
#         print("="*50)


# # Exemplo de uso
# if __name__ == "__main__":
#     # Criar grafo de exemplo
#     print("Criando grafo de exemplo (Les Misérables)...")
#     G = nx.read_graphml('aerial.GraphML')
#     # G = nx.les_miserables_graph()
    
#     print(f"\nGrafo original:")
#     print(f"  Nós: {len(G.nodes())}")
#     print(f"  Arestas: {len(G.edges())}")
    
#     # Aplicar disparity filter
#     print("\nAplicando Disparity Filter...")
#     df = DisparityFilter(G)
    
#     # Computar métricas
#     edges_df = df.compute_filter()
    
#     # Mostrar estatísticas
#     print("\nEstatísticas do filtro:")
#     stats = df.get_summary_statistics()
#     for key, value in stats.items():
#         print(f"  {key}: {value:.4f}")
    
#     # Mostrar quantis
#     df.print_quantiles(num_quantiles=10)
    
#     # Aplicar filtro com alpha = 0.05 (significância de 5%)
#     print(f"\nAplicando corte com alpha = 0.05...")
#     filtered_graph = df.filter_by_alpha(alpha=0.05, min_degree=1)
    
#     print(f"\nGrafo filtrado (alpha=0.05):")
#     print(f"  Nós: {len(filtered_graph.nodes())}")
#     print(f"  Arestas: {len(filtered_graph.edges())}")
#     print(f"  Redução de arestas: {(1 - len(filtered_graph.edges())/len(G.edges()))*100:.1f}%")
    
#     # Aplicar filtro por percentil
#     print(f"\nAplicando corte por percentil (top 30%)...")
#     filtered_graph_ptile = df.filter_by_percentile(percentile=0.70, min_degree=0)
    
#     print(f"\nGrafo filtrado (percentil 70%):")
#     print(f"  Nós: {len(filtered_graph_ptile.nodes())}")
#     print(f"  Arestas: {len(filtered_graph_ptile.edges())}")
#     print(f"  Redução de arestas: {(1 - len(filtered_graph_ptile.edges())/len(G.edges()))*100:.1f}%")
    
#     # Mostrar DataFrame de arestas (primeiras linhas)
#     print("\nPrimeiras arestas do DataFrame:")
#     print(edges_df[['source', 'target', 'weight', 'alpha', 'alpha_percentile']].head(10))

#!/usr/bin/env python
# encoding: utf-8

"""
Disparity Filter para extração de backbone de redes complexas
Baseado em: Serrano et al. (2009) - Extracting the multiscale backbone of complex weighted networks
https://arxiv.org/pdf/0904.2389.pdf
"""

# import networkx as nx
# import numpy as np
# import pandas as pd
# from scipy.stats import percentileofscore
# from typing import Optional, Tuple, Dict
# import warnings
# import h5py


# class DisparityFilter:
#     """
#     Classe para aplicar o Disparity Filter em redes complexas ponderadas.
    
#     O filtro identifica arestas estatisticamente significantes baseado na
#     distribuição de pesos das conexões de cada nó.
#     """
    
#     def __init__(self, graph: nx.Graph):
#         """
#         Inicializa o Disparity Filter com um grafo NetworkX.
        
#         Args:
#             graph: Grafo NetworkX (direcionado ou não) com arestas ponderadas
#         """
#         self.original_graph = graph.copy()
#         self.graph = graph.copy()
#         self.edges_df: Optional[pd.DataFrame] = None
#         self.nodes_df: Optional[pd.DataFrame] = None
#         self.alpha_measures = []
#         self._filter_applied = False
#         self.nodeMapping: Dict = {}
#         self.nodeIDs = list(nx.get_edge_attributes(self.graph, 'nodeId').values())
#         self.nodesToKeep = []
        
#     # NOTA: O método _disparity_integral foi removido, pois a integral 
#     # foi resolvida analiticamente no método abaixo.
    
#     def _get_disparity_significance(self, norm_weight: float, degree: int) -> float:
#         """
#         Calcula o p-value (significância) para o disparity filter.
        
#         Args:
#             norm_weight: Peso normalizado da aresta
#             degree: Grau do nó
            
#         Returns:
#             Valor de alpha (p-value)
#         """
#         # Se o grau for 1, o peso representa 100% da força do nó. 
#         # Estatisticamente, isso não é uma anomalia em relação ao modelo nulo,
#         # portanto o p-value é 1.0 (não significante).
#         if degree <= 1:
#             return 1.0
            
#         # Garante que o peso normalizado fique estritamente entre 0 e 1 
#         # para evitar erros de ponto flutuante na exponenciação
#         norm_weight = min(max(norm_weight, 0.0), 1.0)
            
#         try:
#             # CORREÇÃO: Integral do modelo nulo resolvida analiticamente
#             # Fórmula original: alpha_ij = 1 - integral_0^p_ij (k-1)(1-x)^(k-2) dx
#             # Resultado analítico: alpha_ij = (1 - p_ij)^(k-1)
#             alpha = (1.0 - norm_weight) ** (degree - 1)
#             return alpha
#         except Exception:
#             return 1.0
    
#     def compute_filter(self) -> pd.DataFrame:
#         """
#         Calcula as métricas do disparity filter para todas as arestas.
        
#         Returns:
#             DataFrame com informações detalhadas das arestas incluindo p-values
#         """
#         edges_data = []
#         self.alpha_measures = []
        
#         # Primeira passagem: calcular strength de cada nó
#         node_strength = {}
#         for node_id in self.graph.nodes():
#             strength = 0.0
#             for id0, id1 in self.graph.edges(node_id):
#                 edge_data = self.graph[id0][id1]
#                 weight = edge_data.get('weight', 1.0)
#                 strength += weight
#             node_strength[node_id] = strength
        
#         # Segunda passagem: calcular métricas das arestas
#         for id0, id1 in self.graph.edges():
#             edge_data = self.graph[id0][id1]
#             weight = edge_data.get('weight', 1.0)
            
#             # Informações básicas da aresta
#             edge_info = {
#                 'source': id0,
#                 'target': id1,
#                 'weight': weight
#             }
            
#             # Adicionar coordenadas dos nós se disponíveis
#             if 'pos' in self.graph.nodes[id0]:
#                 edge_info['source_x'] = self.graph.nodes[id0]['pos'][0]
#                 edge_info['source_y'] = self.graph.nodes[id0]['pos'][1]
#             if 'pos' in self.graph.nodes[id1]:
#                 edge_info['target_x'] = self.graph.nodes[id1]['pos'][0]
#                 edge_info['target_y'] = self.graph.nodes[id1]['pos'][1]
            
#             # Calcular alpha para ambas as direções (source e target)
#             degree_source = self.graph.degree(id0)
#             degree_target = self.graph.degree(id1)
            
#             strength_source = node_strength[id0]
#             strength_target = node_strength[id1]
            
#             # Peso normalizado e alpha do source
#             norm_weight_source = weight / strength_source if strength_source > 0 else 0
#             alpha_source = self._get_disparity_significance(norm_weight_source, degree_source)
            
#             # Peso normalizado e alpha do target
#             norm_weight_target = weight / strength_target if strength_target > 0 else 0
#             alpha_target = self._get_disparity_significance(norm_weight_target, degree_target)
            
#             # CORREÇÃO: Usar o MENOR alpha. 
#             # O artigo estipula que a aresta é preservada se for significante 
#             # para pelo menos UM dos nós que ela conecta.
#             alpha = min(alpha_source, alpha_target)
            
#             edge_info.update({
#                 'degree_source': degree_source,
#                 'degree_target': degree_target,
#                 'strength_source': strength_source,
#                 'strength_target': strength_target,
#                 'norm_weight_source': norm_weight_source,
#                 'norm_weight_target': norm_weight_target,
#                 'alpha_source': alpha_source,
#                 'alpha_target': alpha_target,
#                 'alpha': alpha,  # p-value final para uso nos cortes
#             })
            
#             edges_data.append(edge_info)
#             self.alpha_measures.append(alpha)
        
#         # Criar DataFrame
#         self.edges_df = pd.DataFrame(edges_data)
        
#         # Calcular percentis
#         if len(self.alpha_measures) > 0:
#             self.edges_df['alpha_percentile'] = self.edges_df['alpha'].apply(
#                 lambda x: percentileofscore(self.alpha_measures, x) / 100.0
#             )
#         else:
#             self.edges_df['alpha_percentile'] = 0.0
        
#         # Adicionar informações dos nós
#         self._create_nodes_dataframe()
        
#         self._filter_applied = True
#         return self.edges_df
    
#     def _create_nodes_dataframe(self) -> pd.DataFrame:
#         """
#         Cria DataFrame com informações dos nós.
        
#         Returns:
#             DataFrame com informações dos nós
#         """
#         nodes_data = []
        
#         for node_id in self.graph.nodes():
#             node_data = self.graph.nodes[node_id]
#             node_info = {
#                 'node_id': node_id,
#                 'degree': self.graph.degree(node_id),
#             }
            
#             # Adicionar atributos do nó
#             if 'pos' in node_data:
#                 node_info['x'] = node_data['pos'][0]
#                 node_info['y'] = node_data['pos'][1]
            
#             if 'label' in node_data:
#                 node_info['label'] = node_data['label']
            
#             # Adicionar outros atributos personalizados
#             for key, value in node_data.items():
#                 if key not in ['pos', 'label']:
#                     node_info[key] = value
            
#             nodes_data.append(node_info)
        
#         self.nodes_df = pd.DataFrame(nodes_data)
#         return self.nodes_df
    
#     def filter_by_alpha(self, alpha: float, min_degree: int = 1) -> nx.Graph:
#         """
#         Aplica corte no grafo baseado no valor de alpha (p-value).
        
#         Args:
#             alpha: Threshold de significância (p-value). Arestas com alpha >= threshold são removidas
#             min_degree: Grau mínimo para manter nó no grafo
            
#         Returns:
#             Grafo filtrado
#         """
#         if not self._filter_applied:
#             self.compute_filter()
        
#         filtered_graph = self.graph.copy()
        
#         # Remover arestas não significantes
#         edges_to_remove = []
#         for id0, id1 in filtered_graph.edges():
#             edge_mask = (
#                 (self.edges_df['source'] == id0) & 
#                 (self.edges_df['target'] == id1)
#             )
#             edge_row = self.edges_df[edge_mask]
            
#             if not edge_row.empty:
#                 edge_alpha = edge_row['alpha'].values[0]
#                 if edge_alpha >= alpha:  # Remove arestas não significantes
#                     edges_to_remove.append((id0, id1))
        
#         filtered_graph.remove_edges_from(edges_to_remove)
        
#         # Remover nós com grau abaixo do mínimo
#         nodes_to_remove = [
#             node for node in filtered_graph.nodes() 
#             if filtered_graph.degree(node) < min_degree
#         ]
#         self.nodesToKeep = [True if filtered_graph.degree(node) >= min_degree else False for node in filtered_graph.nodes()]
#         filtered_graph.remove_nodes_from(nodes_to_remove)
        
#         return filtered_graph
    
#     def filter_by_percentile(self, percentile: float, min_degree: int = 1) -> nx.Graph:
#         """
#         Aplica corte no grafo baseado no percentil de alpha.
        
#         Args:
#             percentile: Percentil para corte (0.0 a 1.0). Ex: 0.5 mantém top 50% mais significantes
#             min_degree: Grau mínimo para manter nó no grafo
            
#         Returns:
#             Grafo filtrado
#         """
#         if not self._filter_applied:
#             self.compute_filter()
        
#         filtered_graph = self.graph.copy()
        
#         # Remover arestas abaixo do percentil
#         edges_to_remove = []
#         for id0, id1 in filtered_graph.edges():
#             edge_mask = (
#                 (self.edges_df['source'] == id0) & 
#                 (self.edges_df['target'] == id1)
#             )
#             edge_row = self.edges_df[edge_mask]
            
#             if not edge_row.empty:
#                 edge_percentile = edge_row['alpha_percentile'].values[0]
#                 if edge_percentile < percentile:
#                     edges_to_remove.append((id0, id1))
        
#         filtered_graph.remove_edges_from(edges_to_remove)
        
#         # Remover nós com grau abaixo do mínimo
#         nodes_to_remove = [
#             node for node in filtered_graph.nodes() 
#             if filtered_graph.degree(node) < min_degree
#         ]
#         self.nodesToKeep = [True if filtered_graph.degree(node) >= min_degree else False for node in filtered_graph.nodes()]
#         filtered_graph.remove_nodes_from(nodes_to_remove)
        
#         return filtered_graph
    
#     def rebuildH5Data(self, filepath: str, key: str = "df") -> np.ndarray:
#         if not self.nodesToKeep:
#             raise ValueError(
#                 "nodesToKeep vazio. Execute filter_by_alpha/filter_by_percentile antes."
#             )

#         with h5py.File(filepath, "r") as file_obj:
#             if key not in file_obj:
#                 available_keys = list(file_obj.keys())
#                 raise KeyError(
#                     f"Chave '{key}' nao encontrada no H5. Disponiveis: {available_keys}"
#                 )
#             data = np.array(file_obj[key]["block0_values"])

#         if len(self.nodesToKeep) != data.shape[1]:
#             raise ValueError(
#                 "nodesToKeep e quantidade de colunas do H5 nao batem: "
#                 f"{len(self.nodesToKeep)} != {data.shape[1]}"
#             )

#         keep_mask = np.array(self.nodesToKeep, dtype=bool)
#         return data[:, keep_mask]
    
#     def get_edges_dataframe(self) -> pd.DataFrame:
#         """
#         Retorna DataFrame com todas as informações das arestas.
        
#         Returns:
#             DataFrame com arestas e métricas
#         """
#         if not self._filter_applied:
#             self.compute_filter()
#         return self.edges_df
    
#     def get_nodes_dataframe(self) -> pd.DataFrame:
#         """
#         Retorna DataFrame com todas as informações dos nós.
        
#         Returns:
#             DataFrame com nós e atributos
#         """
#         if self.nodes_df is None:
#             self._create_nodes_dataframe()
#         return self.nodes_df
    
#     def get_summary_statistics(self) -> Dict:
#         """
#         Retorna estatísticas resumidas do filtro.
        
#         Returns:
#             Dicionário com estatísticas
#         """
#         if not self._filter_applied:
#             self.compute_filter()
        
#         return {
#             'num_nodes': len(self.graph.nodes()),
#             'num_edges': len(self.graph.edges()),
#             'alpha_mean': np.mean(self.alpha_measures),
#             'alpha_median': np.median(self.alpha_measures),
#             'alpha_std': np.std(self.alpha_measures),
#             'alpha_min': np.min(self.alpha_measures),
#             'alpha_max': np.max(self.alpha_measures),
#         }
    
#     def print_quantiles(self, num_quantiles: int = 10):
#         """
#         Imprime os quantis de alpha para ajudar na escolha do threshold.
        
#         Args:
#             num_quantiles: Número de quantis a calcular
#         """
#         if not self._filter_applied:
#             self.compute_filter()
        
#         bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
#         quantiles = self.edges_df['alpha'].quantile(bins)
        
#         print("\n" + "="*50)
#         print("Quantis de Alpha (p-value)")
#         print("="*50)
#         print(f"{'Percentil':<15} {'Alpha':<15} {'Interpretação'}")
#         print("-"*50)
        
#         for percentile, alpha_val in quantiles.items():
#             interp = "Muito significante" if alpha_val < 0.01 else \
#                      "Significante" if alpha_val < 0.05 else \
#                      "Pouco significante"
#             print(f"{percentile:>6.2%} {alpha_val:>15.4f} {interp}")
        
#         print("="*50)


# # Exemplo de uso
# if __name__ == "__main__":
#     # Criar grafo de exemplo
#     print("Criando grafo de exemplo (Les Misérables)...")
    
#     # Tratamento de erro simples caso o arquivo não exista localmente durante o teste
#     try:
#         G = nx.read_graphml('aerial.GraphML')
#     except FileNotFoundError:
#         print("Arquivo 'aerial.GraphML' não encontrado. Usando Les Misérables para teste.")
#         G = nx.les_miserables_graph()
    
#     print(f"\nGrafo original:")
#     print(f"  Nós: {len(G.nodes())}")
#     print(f"  Arestas: {len(G.edges())}")
    
#     # Aplicar disparity filter
#     print("\nAplicando Disparity Filter...")
#     df = DisparityFilter(G)
    
#     # Computar métricas
#     edges_df = df.compute_filter()
    
#     # Mostrar estatísticas
#     print("\nEstatísticas do filtro:")
#     stats = df.get_summary_statistics()
#     for key, value in stats.items():
#         print(f"  {key}: {value:.4f}")
    
#     # Mostrar quantis
#     df.print_quantiles(num_quantiles=10)
    
#     # Aplicar filtro com alpha = 0.05 (significância de 5%)
#     print(f"\nAplicando corte com alpha = 0.05...")
#     filtered_graph = df.filter_by_alpha(alpha=0.05, min_degree=1)
    
#     print(f"\nGrafo filtrado (alpha=0.05):")
#     print(f"  Nós: {len(filtered_graph.nodes())}")
#     print(f"  Arestas: {len(filtered_graph.edges())}")
#     print(f"  Redução de arestas: {(1 - len(filtered_graph.edges())/len(G.edges()))*100:.1f}%")
    
#     # Aplicar filtro por percentil
#     print(f"\nAplicando corte por percentil (top 30%)...")
#     filtered_graph_ptile = df.filter_by_percentile(percentile=0.70, min_degree=0)
    
#     print(f"\nGrafo filtrado (percentil 70%):")
#     print(f"  Nós: {len(filtered_graph_ptile.nodes())}")
#     print(f"  Arestas: {len(filtered_graph_ptile.edges())}")
#     print(f"  Redução de arestas: {(1 - len(filtered_graph_ptile.edges())/len(G.edges()))*100:.1f}%")
    
#     # Mostrar DataFrame de arestas (primeiras linhas)
#     print("\nPrimeiras arestas do DataFrame:")
#     print(edges_df[['source', 'target', 'weight', 'alpha', 'alpha_percentile']].head(10))

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from typing import Optional, Tuple, Dict
import warnings
import h5py


class DisparityFilter:
    """
    Aplica o Disparity Filter em redes complexas ponderadas.

    O filtro identifica arestas estatisticamente significantes com base na
    distribuição de pesos das conexões de cada nó, conforme proposto por
    Serrano, Boguñá e Vespignani (2009).

    Referência:
        Serrano, M. Á., Boguñá, M., & Vespignani, A. (2009).
        Extracting the multiscale backbone of complex weighted networks.
        PNAS, 106(16), 6483–6488.

    Critério de preservação:
        Uma aresta (i, j) é preservada no backbone se seu p-value (alpha)
        for menor que o limiar α* para pelo menos um dos nós que ela conecta.
        Formalmente: preservar se α_ij(i) < α* OU α_ij(j) < α*.
        Isso equivale a usar min(alpha_source, alpha_target) < α*.

    Notas sobre grafos direcionados:
        Esta implementação foi desenvolvida para grafos não-direcionados,
        seguindo o artigo original. Para grafos direcionados, o cálculo de
        grau e força deveria considerar in-degree e out-degree separadamente.
        O uso com DiGraph é parcialmente suportado (degree retorna in+out),
        mas não foi validado experimentalmente.
    """

    def __init__(self, graph: nx.Graph):
        """
        Inicializa o DisparityFilter com um grafo NetworkX.

        Args:
            graph: Grafo NetworkX (direcionado ou não) com arestas ponderadas.
                   Espera-se que as arestas possuam o atributo 'weight'.
        """
        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.edges_df: Optional[pd.DataFrame] = None
        self.nodes_df: Optional[pd.DataFrame] = None
        self.alpha_measures: list = []
        self._filter_applied: bool = False
        self.nodeMapping: Dict = {}
        self.nodeIDs: list = list(nx.get_edge_attributes(self.graph, 'nodeId').values())
        self.nodesToKeep: list = []

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _get_disparity_significance(self, norm_weight: float, degree: int) -> float:
        """
        Calcula o p-value de uma aresta segundo o modelo nulo do Disparity Filter.

        O modelo nulo assume que os pesos das arestas de um nó de grau k são
        distribuídos uniformemente (distribuição de Dirichlet). O p-value é
        a probabilidade de observar um peso normalizado >= p_ij sob esse modelo.

        Integral resolvida analiticamente:
            alpha_ij = (1 - p_ij)^(k - 1)

        Um p-value baixo (alpha próximo de 0) indica que a aresta é
        estatisticamente improvável sob o modelo nulo — ou seja, ela é
        anômala e deve ser preservada no backbone.

        Args:
            norm_weight: Peso normalizado da aresta (p_ij = w_ij / s_i).
            degree: Grau do nó (número de vizinhos).

        Returns:
            p-value (alpha) no intervalo [0, 1].
        """
        if degree <= 1:
            # Com apenas um vizinho, o peso representa 100% da força.
            # O modelo nulo não é informativo nesse caso → p-value máximo.
            return 1.0

        # Clamp para evitar erros de ponto flutuante na exponenciação.
        norm_weight = min(max(norm_weight, 0.0), 1.0)

        try:
            alpha = (1.0 - norm_weight) ** (degree - 1)
            return float(alpha)
        except Exception:
            return 1.0

    def _compute_node_strength(self) -> Dict:
        """
        Calcula a força (strength) de cada nó: soma dos pesos de suas arestas.

        Returns:
            Dicionário {node_id: strength}.
        """
        node_strength = {}
        for node_id in self.graph.nodes():
            strength = sum(
                self.graph[u][v].get('weight', 1.0)
                for u, v in self.graph.edges(node_id)
            )
            node_strength[node_id] = strength
        return node_strength

    # ------------------------------------------------------------------
    # Computação principal
    # ------------------------------------------------------------------

    def compute_filter(self) -> pd.DataFrame:
        """
        Calcula as métricas do Disparity Filter para todas as arestas.

        Para cada aresta, computa:
        - Grau e força de cada nó endpoint.
        - Peso normalizado de cada perspectiva (source e target).
        - p-value individual por perspectiva (alpha_source, alpha_target).
        - p-value final da aresta: min(alpha_source, alpha_target),
          refletindo o critério de preservação por pelo menos um nó.
        - Percentil do p-value final na distribuição global de alphas.

        Returns:
            DataFrame com uma linha por aresta e todas as métricas calculadas.
        """
        edges_data = []
        self.alpha_measures = []

        node_strength = self._compute_node_strength()

        for id0, id1 in self.graph.edges():
            edge_data = self.graph[id0][id1]
            weight = edge_data.get('weight', 1.0)

            edge_info = {
                'source': id0,
                'target': id1,
                'weight': weight,
            }

            # Coordenadas dos nós, se disponíveis
            for node_id, prefix in [(id0, 'source'), (id1, 'target')]:
                if 'pos' in self.graph.nodes[node_id]:
                    edge_info[f'{prefix}_x'] = self.graph.nodes[node_id]['pos'][0]
                    edge_info[f'{prefix}_y'] = self.graph.nodes[node_id]['pos'][1]

            degree_source = self.graph.degree(id0)
            degree_target = self.graph.degree(id1)
            strength_source = node_strength[id0]
            strength_target = node_strength[id1]

            norm_weight_source = weight / strength_source if strength_source > 0 else 0.0
            norm_weight_target = weight / strength_target if strength_target > 0 else 0.0

            alpha_source = self._get_disparity_significance(norm_weight_source, degree_source)
            alpha_target = self._get_disparity_significance(norm_weight_target, degree_target)

            # Critério do artigo: preservar se significante para PELO MENOS UM nó.
            # Logo, o p-value relevante para decisão de corte é o MENOR dos dois.
            alpha_final = min(alpha_source, alpha_target)

            edge_info.update({
                'degree_source': degree_source,
                'degree_target': degree_target,
                'strength_source': strength_source,
                'strength_target': strength_target,
                'norm_weight_source': norm_weight_source,
                'norm_weight_target': norm_weight_target,
                'alpha_source': alpha_source,
                'alpha_target': alpha_target,
                'alpha': alpha_final,
            })

            edges_data.append(edge_info)
            self.alpha_measures.append(alpha_final)

        self.edges_df = pd.DataFrame(edges_data)

        # Percentil do alpha na distribuição global.
        # Um percentil baixo = alpha baixo = alta significância.
        if len(self.alpha_measures) > 0:
            self.edges_df['alpha_percentile'] = self.edges_df['alpha'].apply(
                lambda x: percentileofscore(self.alpha_measures, x) / 100.0
            )
        else:
            self.edges_df['alpha_percentile'] = 0.0

        self._create_nodes_dataframe()
        self._filter_applied = True
        return self.edges_df

    # ------------------------------------------------------------------
    # Métodos de filtragem
    # ------------------------------------------------------------------

    def filter_by_alpha(self, alpha: float, min_degree: int = 1) -> nx.Graph:
        """
        Filtra o grafo pelo p-value (alpha) do Disparity Filter.

        Segue o critério original do artigo: uma aresta é PRESERVADA se seu
        p-value for MENOR que o limiar α* (i.e., ela é estatisticamente
        significante sob o modelo nulo).

            Preservar  ←→  alpha_ij < α*
            Remover    ←→  alpha_ij >= α*

        Valores típicos de α*: 0.01, 0.05, 0.10.
        Quanto menor o α*, mais restritivo o filtro (menos arestas preservadas).

        Args:
            alpha: Limiar de significância α*. Arestas com p-value >= α* são removidas.
            min_degree: Grau mínimo para manter um nó após a filtragem.

        Returns:
            Novo grafo contendo apenas as arestas do backbone.
        """
        if not self._filter_applied:
            self.compute_filter()

        filtered_graph = self.graph.copy()

        edges_to_remove = [
            (row['source'], row['target'])
            for _, row in self.edges_df.iterrows()
            if row['alpha'] >= alpha  # Remove: não significante (p-value alto)
        ]
        filtered_graph.remove_edges_from(edges_to_remove)

        self.nodesToKeep, nodes_to_remove = self._classify_nodes(filtered_graph, min_degree)
        filtered_graph.remove_nodes_from(nodes_to_remove)

        return filtered_graph

    def filter_by_percentile(self, percentile: float, min_degree: int = 1) -> nx.Graph:
        """
        Filtra o grafo pelo percentil do p-value (alpha) do Disparity Filter.

        Como um alpha baixo indica ALTA significância, preservar as arestas
        mais significantes equivale a manter aquelas com alpha no percentil
        INFERIOR da distribuição.

        O parâmetro `percentile` define o limiar superior de alpha_percentile
        para preservação. Portanto:

            Preservar  ←→  alpha_percentile <= percentile
            Remover    ←→  alpha_percentile > percentile

        Exemplos:
            percentile=0.30 → mantém as 30% de arestas com menor alpha
                              (as mais significantes).
            percentile=0.50 → mantém a metade mais significante do grafo.

        Nota: Esta semântica é inversa à de um corte por peso bruto,
        onde valores altos seriam preservados. Aqui, valores BAIXOS de alpha
        são os desejáveis.

        Args:
            percentile: Fração de arestas a preservar (0.0 a 1.0),
                        selecionando as de menor alpha (maior significância).
            min_degree: Grau mínimo para manter um nó após a filtragem.

        Returns:
            Novo grafo contendo apenas as arestas do backbone.
        """
        if not self._filter_applied:
            self.compute_filter()

        filtered_graph = self.graph.copy()

        # CORREÇÃO: remover arestas com alpha_percentile ALTO (pouco significantes).
        # Na versão anterior, a lógica estava invertida — removia as mais significantes.
        edges_to_remove = [
            (row['source'], row['target'])
            for _, row in self.edges_df.iterrows()
            if row['alpha_percentile'] > percentile  # Remove: pouco significante
        ]
        filtered_graph.remove_edges_from(edges_to_remove)

        self.nodesToKeep, nodes_to_remove = self._classify_nodes(filtered_graph, min_degree)
        filtered_graph.remove_nodes_from(nodes_to_remove)

        return filtered_graph

    # ------------------------------------------------------------------
    # Utilitários internos de filtragem
    # ------------------------------------------------------------------

    def _classify_nodes(
        self, filtered_graph: nx.Graph, min_degree: int
    ) -> Tuple[list, list]:
        """
        Classifica nós como a manter ou remover com base no grau mínimo.

        Args:
            filtered_graph: Grafo já com arestas filtradas.
            min_degree: Grau mínimo para preservação do nó.

        Returns:
            Tupla (nodesToKeep, nodes_to_remove), onde nodesToKeep é uma lista
            de booleanos alinhada com a ordem dos nós do grafo original, e
            nodes_to_remove é a lista de IDs dos nós a eliminar.
        """
        nodes_to_keep_flags = []
        nodes_to_remove = []

        for node in self.graph.nodes():  # Itera sobre o grafo ORIGINAL para manter alinhamento
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
                'node_id': node_id,
                'degree': self.graph.degree(node_id),
            }
            if 'pos' in node_data:
                node_info['x'] = node_data['pos'][0]
                node_info['y'] = node_data['pos'][1]
            if 'label' in node_data:
                node_info['label'] = node_data['label']
            for key, value in node_data.items():
                if key not in ('pos', 'label'):
                    node_info[key] = value
            nodes_data.append(node_info)

        self.nodes_df = pd.DataFrame(nodes_data)
        return self.nodes_df

    def get_edges_dataframe(self) -> pd.DataFrame:
        """
        Retorna DataFrame com todas as métricas das arestas.

        Returns:
            DataFrame com arestas e respectivos p-values e percentis.
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
    # Reconstrução de dados H5
    # ------------------------------------------------------------------

    def rebuildH5Data(self, filepath: str, key: str = "df") -> np.ndarray:
        """
        Reconstrói o array de séries temporais removendo nós eliminados pelo filtro.

        A correspondência entre nodesToKeep e colunas do H5 assume que a ordem
        dos nós no grafo original coincide com a ordem das colunas no arquivo H5.
        Certifique-se de que o grafo foi construído preservando essa ordem.

        Args:
            filepath: Caminho para o arquivo H5.
            key: Chave do grupo no arquivo H5 (padrão: "df").

        Returns:
            Array numpy com shape (T, N_kept), onde N_kept é o número de nós
            preservados após a filtragem.

        Raises:
            ValueError: Se nodesToKeep estiver vazio ou incompatível com o H5.
            KeyError: Se a chave não existir no arquivo H5.
        """
        if not self.nodesToKeep:
            raise ValueError(
                "nodesToKeep está vazio. Execute filter_by_alpha() ou "
                "filter_by_percentile() antes de reconstruir os dados H5."
            )

        with h5py.File(filepath, "r") as f:
            if key not in f:
                available = list(f.keys())
                raise KeyError(
                    f"Chave '{key}' não encontrada no arquivo H5. "
                    f"Chaves disponíveis: {available}"
                )
            data = np.array(f[key]["block0_values"])

        if len(self.nodesToKeep) != data.shape[1]:
            raise ValueError(
                f"Incompatibilidade entre nodesToKeep ({len(self.nodesToKeep)}) "
                f"e número de colunas no H5 ({data.shape[1]}). "
                "Verifique se o grafo foi construído com a mesma ordem dos nós do arquivo H5."
            )

        keep_mask = np.array(self.nodesToKeep, dtype=bool)
        return data[:, keep_mask]

    # ------------------------------------------------------------------
    # Estatísticas e diagnóstico
    # ------------------------------------------------------------------

    def get_summary_statistics(self) -> Dict:
        """
        Retorna estatísticas descritivas da distribuição de p-values das arestas.

        Returns:
            Dicionário com contagens do grafo e estatísticas de alpha.
        """
        if not self._filter_applied:
            self.compute_filter()

        return {
            'num_nodes': len(self.graph.nodes()),
            'num_edges': len(self.graph.edges()),
            'alpha_mean': float(np.mean(self.alpha_measures)),
            'alpha_median': float(np.median(self.alpha_measures)),
            'alpha_std': float(np.std(self.alpha_measures)),
            'alpha_min': float(np.min(self.alpha_measures)),
            'alpha_max': float(np.max(self.alpha_measures)),
        }

    def backbone_report(self, alpha: float) -> Dict:
        """
        Gera um relatório comparativo entre o grafo original e o backbone
        extraído com um dado limiar α*.

        Útil para avaliar o impacto do corte antes de aplicá-lo ao pipeline.

        Args:
            alpha: Limiar α* a ser avaliado.

        Returns:
            Dicionário com contagens e taxas de redução.
        """
        if not self._filter_applied:
            self.compute_filter()

        n_original_edges = len(self.graph.edges())
        n_significant = (self.edges_df['alpha'] < alpha).sum()
        n_removed = n_original_edges - n_significant

        return {
            'alpha_threshold': alpha,
            'original_edges': n_original_edges,
            'backbone_edges': int(n_significant),
            'removed_edges': int(n_removed),
            'edge_retention_rate': round(n_significant / n_original_edges, 4) if n_original_edges > 0 else 0.0,
            'edge_removal_rate': round(n_removed / n_original_edges, 4) if n_original_edges > 0 else 0.0,
        }

    def print_quantiles(self, num_quantiles: int = 10):
        """
        Exibe os quantis da distribuição de p-values (alpha) para auxiliar
        na escolha do limiar de corte.

        Lembrete de interpretação:
            - Alpha baixo → aresta muito significante → deve ser PRESERVADA.
            - Alpha alto  → aresta pouco significante → candidata à REMOÇÃO.

        Args:
            num_quantiles: Número de quantis a exibir.
        """
        if not self._filter_applied:
            self.compute_filter()

        bins = np.linspace(0, 1, num=num_quantiles + 1, endpoint=True)
        quantiles = self.edges_df['alpha'].quantile(bins)

        print("\n" + "=" * 60)
        print("Distribuição de p-values (alpha) — Disparity Filter")
        print("Lembrete: alpha BAIXO = aresta SIGNIFICANTE = PRESERVAR")
        print("=" * 60)
        print(f"{'Percentil':<15} {'Alpha':<15} {'Interpretação'}")
        print("-" * 60)

        for percentile_val, alpha_val in quantiles.items():
            if alpha_val < 0.01:
                interp = "Muito significante → preservar"
            elif alpha_val < 0.05:
                interp = "Significante → preservar"
            elif alpha_val < 0.10:
                interp = "Limítrofe"
            else:
                interp = "Pouco significante → remover"
            print(f"{percentile_val:>6.2%}         {alpha_val:>10.4f}     {interp}")

        print("=" * 60)


# ------------------------------------------------------------------
# Exemplo de uso
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Criando grafo de exemplo...")
    alpha = 0.4

    try:
        # G = nx.read_graphml('aerial.GraphML')
        G = nx.read_graphml("../data/GraphML/metr-la.GraphML") 
        
    except FileNotFoundError:
        print("Arquivo 'aerial.GraphML' não encontrado. Usando Les Misérables para teste.")
        G = nx.les_miserables_graph()

    print(f"\nGrafo original: {len(G.nodes())} nós, {len(G.edges())} arestas")

    df_filter = DisparityFilter(G)
    edges_df = df_filter.compute_filter()

    print("\nEstatísticas do filtro:")
    for k, v in df_filter.get_summary_statistics().items():
        print(f"  {k}: {v}")

    df_filter.print_quantiles(num_quantiles=10)

    # Corte por alpha = 0.05 (preserva arestas com p-value < 0.05)
    print(f"\nAplicando corte com alpha = {alpha}...")
    report = df_filter.backbone_report(alpha=alpha)
    for k, v in report.items():
        print(f"  {k}: {v}")

    backbone_alpha = df_filter.filter_by_alpha(alpha=alpha, min_degree=1)
    print(f"\nBackbone (alpha={alpha}): {len(backbone_alpha.nodes())} nós, {len(backbone_alpha.edges())} arestas")

    # Corte por percentil: mantém as 30% de arestas com menor alpha (mais significantes)
    print("\nAplicando corte por percentil (30% mais significantes)...")
    backbone_ptile = df_filter.filter_by_percentile(percentile=0.30, min_degree=1)
    print(f"Backbone (percentile=0.30): {len(backbone_ptile.nodes())} nós, {len(backbone_ptile.edges())} arestas")

    print("\nPrimeiras arestas com métricas:")
    print(edges_df[['source', 'target', 'weight', 'alpha', 'alpha_percentile']].head(10))
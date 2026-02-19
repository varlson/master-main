"""
Utilitários para carregamento e preprocessamento de dados
"""

import numpy as np
import h5py
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, Optional
import networkx as nx

from config import *


class TrafficDataset(Dataset):
    """Dataset personalizado para dados de tráfego"""
    
    def __init__(self, X, Y):
        """
        Args:
            X: Sequências de entrada [num_samples, seq_len, num_nodes, features]
            Y: Sequências de saída [num_samples, horizon, num_nodes, features]
        """
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_h5_data(filepath: Path) -> np.ndarray:
    """
    Carrega dados de arquivo H5
    
    Args:
        filepath: Caminho para o arquivo .h5
        key: Chave do dataset dentro do H5
        
    Returns:
        Array numpy com os dados
    """
    # with h5py.File(filepath, 'r') as f:
    #     if key not in f.keys():
    #         available_keys = list(f.keys())
    #         raise KeyError(f"Chave '{key}' não encontrada. Chaves disponíveis: {available_keys}")
    #     data = np.array(f[key]['block0_values'])

    
    data = np.load(filepath, allow_pickle=True)
    print(f"✓ H5 Dados carregados de {filepath.name}: shape {data.shape}")
    return data


def load_pickle_adj_matrix(filepath: Path) -> np.ndarray:
    """
    Carrega matriz de adjacência de arquivo pickle
    
    Args:
        filepath: Caminho para o arquivo .pkl
        
    Returns:
        Matriz de adjacência
    """
    
    
    with open(filepath, 'rb') as f:
        try:
            if "with" in filepath:
                adj_mx = pickle.load(f, encoding='latin1')
            else:
                _, _, adj_mx = pickle.load(f, encoding='latin1')
        except:
            adj_mx = pickle.load(f, encoding='latin1')
    
    # Converter para denso se necessário
    if sp.issparse(adj_mx):
        adj_mx = adj_mx.toarray()
    
    # print(f"✓ Matriz de adjacência carregada de {filepath.name}: shape {adj_mx.shape}")
    return adj_mx


def load_graphml_backbone(filepath: Path) -> np.ndarray:
    """
    Carrega backbone de arquivo GraphML
    
    Args:
        filepath: Caminho para o arquivo .GraphML
        
    Returns:
        Matriz de adjacência do backbone
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo GraphML não encontrado: {filepath}")
    
    g = nx.read_graphml(filepath)
    backbone_adj = nx.adjacency_matrix(g).toarray()
    
    # print(f"✓ Backbone carregado de {filepath.name}: shape {backbone_adj.shape}")
    return backbone_adj


def normalize_data(data: np.ndarray, 
                   method: str = "zscore") -> Tuple[np.ndarray, dict]:
    """
    Normaliza os dados
    
    Args:
        data: Dados a normalizar
        method: Método de normalização ('zscore', 'minmax')
        
    Returns:
        Dados normalizados e dicionário com estatísticas
    """
    if method == "zscore":
        mean = np.mean(data)
        std = np.std(data)
        normalized = (data - mean) / (std + 1e-8)
        stats = {"mean": mean, "std": std, "method": "zscore"}
        
    elif method == "minmax":
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        stats = {"min": min_val, "max": max_val, "method": "minmax"}
    
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    return normalized, stats


def create_sequences(data: np.ndarray, 
                     seq_len: int, 
                     horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria sequências para previsão temporal
    
    Args:
        data: Dados temporais [timesteps, num_nodes, features]
        seq_len: Comprimento da janela de entrada
        horizon: Horizonte de previsão
        
    Returns:
        X: Sequências de entrada [num_samples, seq_len, num_nodes, features]
        Y: Sequências de saída [num_samples, horizon, num_nodes, features]
    """
    num_samples = len(data) - seq_len - horizon + 1
    
    X, Y = [], []
    for i in range(num_samples):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+horizon])
    
    return np.array(X), np.array(Y)


def split_data(data: np.ndarray, 
               train_ratio: float = 0.7,
               val_ratio: float = 0.1,
               test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide dados em treino, validação e teste
    
    Args:
        data: Dados completos
        train_ratio: Proporção de treino
        val_ratio: Proporção de validação
        test_ratio: Proporção de teste
        
    Returns:
        train_data, val_data, test_data
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "As proporções devem somar 1.0"
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"✓ Divisão dos dados:")
    print(f"  - Treino: {len(train_data)} amostras")
    print(f"  - Validação: {len(val_data)} amostras")
    print(f"  - Teste: {len(test_data)} amostras")
    
    return train_data, val_data, test_data


def prepare_dataloaders(
    data_file: Path,
    adj_file: Path,
    seq_len: int = 12,
    horizon: int = 12,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    backbone_adj_mx: Optional[np.ndarray] = None,
    normalize: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, int, np.ndarray, dict]:
    """
    Prepara dataloaders para treino, validação e teste
    
    Args:
        data_file: Arquivo H5 com os dados
        adj_file: Arquivo PKL com matriz de adjacência
        dataset_key: Chave do dataset no H5
        seq_len: Comprimento da sequência de entrada
        horizon: Horizonte de previsão
        batch_size: Tamanho do batch
        train_ratio: Proporção de treino
        val_ratio: Proporção de validação
        test_ratio: Proporção de teste
        backbone_adj_mx: Matriz de adjacência do backbone (opcional)
        normalize: Se True, normaliza os dados
        num_workers: Número de workers para o DataLoader
        pin_memory: Se True, usa pin_memory (para GPU)
        
    Returns:
        train_loader, val_loader, test_loader, num_nodes, adj_mx, normalization_stats
    """
    print(f"\n{'='*60}")
    print(f"Preparando dataloaders para {data_file.name}")
    print(f"{'='*60}")
    
    # Carregar dados
    
    
    data = load_h5_data(data_file)

    
    adj_mx = load_pickle_adj_matrix(adj_file)
    
    # Usar backbone se fornecido
    if backbone_adj_mx is not None:
        print("✓ Usando backbone fornecido para matriz de adjacência")
        adj_mx = backbone_adj_mx
    print(f"✓ Matriz de adjacência carregada de {adj_file.name}: shape {adj_mx.shape}")
    
    
    # Obter número de nós
    num_nodes = adj_mx.shape[0]
    
    # Garantir formato correto [timesteps, num_nodes, features]
    if data.ndim == 2:
        data = data[:, :, np.newaxis]  # Adiciona dimensão de features
    
    # Normalizar dados
    normalization_stats = {}
    if normalize:
        data, normalization_stats = normalize_data(data, method="zscore")
        print(f"✓ Dados normalizados (Z-score): mean={normalization_stats['mean']:.2f}, "
              f"std={normalization_stats['std']:.2f}")
    
    # Dividir dados
    train_data, val_data, test_data = split_data(
        data, train_ratio, val_ratio, test_ratio
    )
    
    # Criar sequências
    X_train, Y_train = create_sequences(train_data, seq_len, horizon)
    X_val, Y_val = create_sequences(val_data, seq_len, horizon)
    X_test, Y_test = create_sequences(test_data, seq_len, horizon)
    
    print(f"✓ Sequências criadas:")
    print(f"  - X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  - X_val: {X_val.shape}, Y_val: {Y_val.shape}")
    print(f"  - X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    
    # Criar datasets
    train_dataset = TrafficDataset(X_train, Y_train)
    val_dataset = TrafficDataset(X_val, Y_val)
    test_dataset = TrafficDataset(X_test, Y_test)
    
    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    print(f"✓ DataLoaders criados com batch_size={batch_size}")
    print(f"{'='*60}\n")
    
    return (train_loader, val_loader, test_loader, 
            num_nodes, adj_mx, normalization_stats)


def denormalize_predictions(predictions: np.ndarray, 
                            stats: dict) -> np.ndarray:
    """
    Desnormaliza predições usando estatísticas salvas
    
    Args:
        predictions: Predições normalizadas
        stats: Dicionário com estatísticas de normalização
        
    Returns:
        Predições desnormalizadas
    """
    if stats.get("method") == "zscore":
        return predictions * stats["std"] + stats["mean"]
    elif stats.get("method") == "minmax":
        return predictions * (stats["max"] - stats["min"]) + stats["min"]
    else:
        return predictions
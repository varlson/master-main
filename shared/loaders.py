
from typing import Optional, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset):
    """Dataset personalizado para dados de tráfego"""

    def __init__(self, X, Y):
        """
        Args:
            X: [num_samples, seq_len, num_nodes, features]
            Y: [num_samples, horizon, num_nodes, features]
        """
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def ensure_data_shape(data: np.ndarray) -> np.ndarray:
    # (T, N) -> (T, N, 1)
    if data.ndim == 2:
        return data[:, :, np.newaxis]
    if data.ndim == 3:
        return data
    raise ValueError(f"data deve ter shape (T,N) ou (T,N,F). Recebido: {data.shape}")


def ensure_adj_shape(adj_mx: np.ndarray) -> np.ndarray:
    if adj_mx.ndim != 2 or adj_mx.shape[0] != adj_mx.shape[1]:
        raise ValueError(f"adj_mx deve ser quadrada (N,N). Recebido: {adj_mx.shape}")
    return adj_mx


def split_data(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "As proporções devem somar 1.0"

    n = data.shape[0]
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return data[:train_end], data[train_end:val_end], data[val_end:]


def create_sequences(
    data: np.ndarray,
    seq_len: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    T = data.shape[0]
    num_samples = T - seq_len - horizon + 1
    if num_samples <= 0:
        raise ValueError(
            f"Dados insuficientes: T={T}, seq_len={seq_len}, horizon={horizon}."
        )

    X, Y = [], []
    for i in range(num_samples):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + horizon])

    return np.array(X), np.array(Y)


def compute_normalization_stats(train_data: np.ndarray, method: str = "zscore") -> Dict:
    if method == "zscore":
        mean = float(np.mean(train_data))
        std = float(np.std(train_data))
        return {"method": "zscore", "mean": mean, "std": std}

    if method == "minmax":
        min_val = float(np.min(train_data))
        max_val = float(np.max(train_data))
        return {"method": "minmax", "min": min_val, "max": max_val}

    raise ValueError(f"Método desconhecido: {method}")


def apply_normalization(data: np.ndarray, stats: Dict) -> np.ndarray:
    if stats["method"] == "zscore":
        return (data - stats["mean"]) / (stats["std"] + 1e-8)

    if stats["method"] == "minmax":
        return (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)

    raise ValueError(f"Stats inválidas: {stats}")


def prepare_dataloaders_from_arrays(
    data: np.ndarray,                      # ex: (34272, 207) ou (34272, 207, 1)
    adj_mx: np.ndarray,                    # ex: (207, 207)
    seq_len: int = 12,
    horizon: int = 12,
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    backbone_adj_mx: Optional[np.ndarray] = None,
    normalize: bool = True,
    normalization_method: str = "zscore",
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, int, np.ndarray, dict]:

    print(f"\n{'='*60}")
    print("Preparando dataloaders (arrays em memória)")
    print(f"{'='*60}")

    data = ensure_data_shape(data)
    adj_mx = ensure_adj_shape(adj_mx)

    if backbone_adj_mx is not None:
        print("✓ Usando backbone fornecido para matriz de adjacência")
        backbone_adj_mx = ensure_adj_shape(backbone_adj_mx)
        adj_mx = backbone_adj_mx

    num_nodes = adj_mx.shape[0]
    print(f"✓ Matriz de adjacência: shape {adj_mx.shape}")
    print(f"✓ Dados: shape {data.shape}")

    if data.shape[1] != num_nodes:
        raise ValueError(
            f"Incompatibilidade: data tem {data.shape[1]} nós, adj_mx tem {num_nodes}. "
            f"data.shape={data.shape}, adj_mx.shape={adj_mx.shape}"
        )

    # split
    train_data, val_data, test_data = split_data(data, train_ratio, val_ratio, test_ratio)
    print("✓ Divisão dos dados:")
    print(f"  - Treino: {train_data.shape[0]} amostras")
    print(f"  - Validação: {val_data.shape[0]} amostras")
    print(f"  - Teste: {test_data.shape[0]} amostras")

    # normalize (fit no treino)
    normalization_stats = {}
    if normalize:
        normalization_stats = compute_normalization_stats(train_data, method=normalization_method)
        train_data = apply_normalization(train_data, normalization_stats)
        val_data = apply_normalization(val_data, normalization_stats)
        test_data = apply_normalization(test_data, normalization_stats)

        if normalization_stats["method"] == "zscore":
            print(f"✓ Normalização Z-score: mean={normalization_stats['mean']:.4f}, "
                  f"std={normalization_stats['std']:.4f}")
        else:
            print(f"✓ Normalização MinMax: min={normalization_stats['min']:.4f}, "
                  f"max={normalization_stats['max']:.4f}")

    # sequences
    X_train, Y_train = create_sequences(train_data, seq_len, horizon)
    X_val, Y_val = create_sequences(val_data, seq_len, horizon)
    X_test, Y_test = create_sequences(test_data, seq_len, horizon)

    print("✓ Sequências criadas:")
    print(f"  - X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"  - X_val:   {X_val.shape},   Y_val:   {Y_val.shape}")
    print(f"  - X_test:  {X_test.shape},  Y_test:  {Y_test.shape}")

    # datasets (seu TrafficDataset converte pra torch.FloatTensor)
    train_dataset = TrafficDataset(X_train, Y_train)
    val_dataset = TrafficDataset(X_val, Y_val)
    test_dataset = TrafficDataset(X_test, Y_test)

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

    return train_loader, val_loader, test_loader, num_nodes, adj_mx, normalization_stats




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
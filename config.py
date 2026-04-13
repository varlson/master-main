"""
Configuração central do pipeline de experimentos GNN
"""

from pathlib import Path

import torch

# ============================================
# PATHS E DIRETÓRIOS
# ============================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
H5_DIR = DATA_DIR / "h5"
NPY_DIR = DATA_DIR / "npy"
PKL_DIR = DATA_DIR / "pkl"
GRAPHML_DIR = BASE_DIR / "dataset" / "GraphML"
RESULTS_DIR = BASE_DIR / "results"
MLFLOW_DIR = BASE_DIR / "mlruns"
PERCENTILE = 0.30
MIN_DEGREE  = 1
ALPHA = 0.3

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, H5_DIR, PKL_DIR, GRAPHML_DIR, RESULTS_DIR, MLFLOW_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================
# CONFIGURAÇÕES DE HARDWARE
# ============================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4
PIN_MEMORY = True if DEVICE == 'cuda' else False


# ============================================
# CONFIGURAÇÕES DE DADOS
# ============================================

# Divisão dos dados
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# Sequências temporais
SEQ_LEN = 12  # Janela de entrada (12 timesteps = 1 hora com dados de 5min)
HORIZON = 12  # Horizonte de previsão

# Batch size
BATCH_SIZE = 64


# ============================================
# DATASETS DISPONÍVEIS
# ============================================

DATASET_LIST = ["metr-la", "pems-bay"]

DATASETS = {
    "metr-la": {
        "npy_file": "metr-la.npy",
        "pkl_file": "metr-la.pkl",
        "description": "Los Angeles Metro Traffic",
        "num_nodes": 207,
        "feature": "speed"
    },
    "pems-bay": {
        "npy_file": "pems-bay.npy",
        "pkl_file": "pems-bay.pkl",
        "description": "Bay Area Traffic",
        "num_nodes": 325,
        "feature": "speed"
    },
    # Template para adicionar novos datasets
    # "novo-dataset": {
    #     "h5_file": "NOVO-DATASET.h5",
    #     "pkl_file": "adj_mx_novo.pkl",
    #     "key": "data",
    #     "description": "Descrição do dataset",
    #     "num_nodes": None,  # será detectado automaticamente
    #     "feature": "speed"
    # }
}


# ============================================
# CONFIGURAÇÕES DE BACKBONE
# ============================================



BACKBONE_METHODS = {
    "disp_fil": [ 0.2],
    "nois_corr": [0.7],
    # "threshold": [0.1, 0.2, 0.3, 0.4, 0.5]
}


# ============================================
# CONFIGURAÇÕES DE TREINAMENTO
# ============================================

# Número de épocas (pode ser sobrescrito no grid search)
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# MLflow
MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR}"
MLFLOW_EXPERIMENT_PREFIX = "GNN_Traffic_Forecasting"


# ============================================
# GRID SEARCH - MODELOS
# ============================================

GRID_SEARCH_CONFIGS = {
    "GraphWaveNet": {
        "input_dim": [1],
        "output_dim": [1],
        "seq_len": [SEQ_LEN],
        "horizon": [HORIZON],
        "hidden_dim": [32, 64, 128],
        "num_blocks": [3, 4, 5],
        "dilation_base": [2],
        "k": [2, 3],
        "dropout": [0.1, 0.2],
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-3, 1e-4],
        "epochs": [EPOCHS],
        "patience": [EARLY_STOPPING_PATIENCE]
    },
    
    "DCRNN": {
        "input_dim": [1],
        "output_dim": [1],
        "seq_len": [SEQ_LEN],
        "horizon": [HORIZON],
        "hidden_dim": [32, 64, 128],
        "k": [2, 3],
        "dropout": [0.1, 0.2],
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-3, 1e-4],
        "epochs": [EPOCHS],
        "patience": [EARLY_STOPPING_PATIENCE],
        "use_scheduled_sampling": [False, True],
        "teacher_forcing_ratio": [0.5]
    },
    
    # Template para novos modelos
    # "NovoModelo": {
    #     "input_dim": [1],
    #     "output_dim": [1],
    #     "seq_len": [SEQ_LEN],
    #     "horizon": [HORIZON],
    #     # ... outros hiperparâmetros
    # }
}


# ============================================
# CONFIGURAÇÕES DE TESTES RÁPIDOS (DEBUG)
# ============================================

DEBUG_MODE = False  # Ativar para testes rápidos

if DEBUG_MODE:
    EPOCHS = 2
    BATCH_SIZE = 8
    
    # Grid search reduzido para debug
    GRID_SEARCH_CONFIGS = {
        "GraphWaveNet": {
            "input_dim": [1],
            "output_dim": [1],
            "seq_len": [SEQ_LEN],
            "horizon": [HORIZON],
            "hidden_dim": [64],
            "num_blocks": [3],
            "dilation_base": [2],
            "k": [2],
            "dropout": [0.1],
            "lr": [1e-3],
            "weight_decay": [1e-3],
            "epochs": [2],
            "patience": [2]
        },
        "DCRNN": {
            "input_dim": [1],
            "output_dim": [1],
            "seq_len": [SEQ_LEN],
            "horizon": [HORIZON],
            "hidden_dim": [64],
            "k": [2],
            "dropout": [0.1],
            "lr": [1e-3],
            "weight_decay": [1e-3],
            "epochs": [2],
            "patience": [2],
            "use_scheduled_sampling": [False],
            "teacher_forcing_ratio": [0.5]
        }
    }


# ============================================
# MÉTRICAS DE AVALIAÇÃO
# ============================================

PRIMARY_METRIC = "MAE"  # Métrica principal para seleção de modelos
METRICS = ["MAE", "RMSE", "MAPE", "Loss"]


# ============================================
# CONFIGURAÇÕES DE LOGGING
# ============================================

VERBOSE = True
SAVE_PREDICTIONS = False  # Salvar predições para análise posterior
SAVE_MODELS = True  # Salvar checkpoints dos melhores modelos


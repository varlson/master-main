import sys
import os
sys.path.append(os.path.abspath(".."))

import time
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from shared import prepare_dataloaders_from_arrays
from models.WaveNet import GraphWaveNet

# 1. Configuração de Caminhos e Dispositivo
# Ajustando caminhos para subir um nível a partir de /_lab
data_path = "../data/npy/pems-bay-h5.npy"
adj_path = "../data/npy/pems-bay-adj_mx.npy"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- DIAGNÓSTICO DE PERFORMANCE ---")
print(f"Device detectado: {device}")

# 2. Carga de Dados
print("\nCarregando dados...")
start_data = time.time()
data = np.load(data_path)
adj = np.load(adj_path)
print(f"Dados carregados em {time.time() - start_data:.2f}s")

# 3. Preparação de DataLoaders
print("\nPreparando dataloaders...")
train_loader, val_loader, test_loader, num_nodes, adj_mx, normalization_stats = (
    prepare_dataloaders_from_arrays(
        data=data,
        adj_mx=adj,
        seq_len=12,
        horizon=12,
        batch_size=128,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        normalize=True,
        normalization_method='zscore',
        num_workers=4,
        pin_memory=(device == "cuda"),
    )
)

# 4. Instanciação do Modelo
common_kwargs = {
    "adj_mx": adj_mx,
    "num_nodes": num_nodes,
    "input_dim": 1,
    "hidden_dim": 64,
    "output_dim": 1,
    "seq_len": 12,
    "horizon": 12,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 1,  # Apenas 1 época para diagnóstico rápido
    "patience": 5,
    "device": device,
}

model = GraphWaveNet(
    **common_kwargs,
    num_blocks=3,
    dilation_base=2,
    k=2,
)

# 5. Treinamento com Profiling
print("\n--- INICIANDO TREINAMENTO (1 ÉPOCA) ---")
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

start_train = time.time()

# Usando o profiler nativo do PyTorch para identificar gargalos
activities = [ProfilerActivity.CPU]
if device == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=True) as prof:
    with record_function("model_training"):
        model.fit(train_loader, val_loader)

train_duration = time.time() - start_train
print(f"\nTreinamento concluído em: {train_duration:.2f}s")

if device == "cuda":
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Pico de memória GPU: {max_mem:.2f} MB")

# 6. Avaliação (Teste) Cronometrada
print("\n--- INICIANDO TESTE (AVALIAÇÃO) ---")
start_test = time.time()
test_loss = model.evaluate(test_loader)
test_duration = time.time() - start_test
print(f"Avaliação concluída em: {test_duration:.2f}s | Loss: {test_loss:.4f}")

# 7. Resumo do Profiler (Top 10 operações)
print("\n--- TOP 10 OPERAÇÕES (CPU/GPU) ---")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

if device == "cuda":
    print("\n--- TOP 10 OPERAÇÕES (GPU KERNELS) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
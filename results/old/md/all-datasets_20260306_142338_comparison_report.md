# Relatório de Comparação de Experimentos GNN

**Gerado em:** 2026-03-06 19:12:47

**Total de experimentos:** 12

## Resumo por Modelo

### DCRNN

- Experimentos: 2
- MAE médio: 0.3135 ± 0.0389
- RMSE médio: 0.5807 ± 0.0647
- MAPE médio: 146.47% ± 32.84%

### GraphWaveNet

- Experimentos: 2
- MAE médio: 0.2704 ± 0.0780
- RMSE médio: 0.5367 ± 0.1187
- MAPE médio: 117.31% ± 5.87%

### MTGNN

- Experimentos: 2
- MAE médio: 0.2967 ± 0.0795
- RMSE médio: 0.5716 ± 0.0902
- MAPE médio: 129.72% ± 7.98%

### DGCRN

- Experimentos: 2
- MAE médio: 0.2760 ± 0.0448
- RMSE médio: 0.5644 ± 0.0906
- MAPE médio: 144.35% ± 28.51%

### STICformer

- Experimentos: 2
- MAE médio: 0.2566 ± 0.0749
- RMSE médio: 0.5211 ± 0.1242
- MAPE médio: 118.67% ± 7.01%

### PatchSTG

- Experimentos: 2
- MAE médio: 0.2579 ± 0.0712
- RMSE médio: 0.5218 ± 0.1228
- MAPE médio: 117.41% ± 8.09%

## Melhores Resultados por Dataset

### pems-bay

- **Melhor modelo:** STICformer
- **Experimento:** pems-bay_STICformer_20260306_142338
- **MAE:** 0.2036
- **RMSE:** 0.4333
- **MAPE:** 123.63%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'num_layers': 2, 'num_heads': 4, 'ff_multiplier': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 200, 'patience': 5}`

### metr-la

- **Melhor modelo:** DGCRN
- **Experimento:** metr-la_DGCRN_20260306_142338
- **MAE:** 0.3077
- **RMSE:** 0.6284
- **MAPE:** 124.19%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'node_dim': 16, 'gcn_depth': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 200, 'patience': 5}`

## Tabela Completa de Resultados

| Experimento | Modelo | Dataset | MAE | RMSE | MAPE (%) |
|-------------|--------|---------|-----|------|----------|
| pems-bay_DCRNN_20260306_142338 | DCRNN | pems-bay | 0.2860 | 0.5349 | 169.69 |
| pems-bay_GraphWaveNet_20260306_142338 | GraphWaveNet | pems-bay | 0.2152 | 0.4528 | 121.46 |
| pems-bay_MTGNN_20260306_142338 | MTGNN | pems-bay | 0.2405 | 0.5078 | 135.36 |
| pems-bay_DGCRN_20260306_142338 | DGCRN | pems-bay | 0.2443 | 0.5003 | 164.52 |
| pems-bay_STICformer_20260306_142338 | STICformer | pems-bay | 0.2036 | 0.4333 | 123.63 |
| pems-bay_PatchSTG_20260306_142338 | PatchSTG | pems-bay | 0.2075 | 0.4350 | 123.13 |
| metr-la_DCRNN_20260306_142338 | DCRNN | metr-la | 0.3411 | 0.6264 | 123.25 |
| metr-la_GraphWaveNet_20260306_142338 | GraphWaveNet | metr-la | 0.3256 | 0.6207 | 113.15 |
| metr-la_MTGNN_20260306_142338 | MTGNN | metr-la | 0.3529 | 0.6354 | 124.08 |
| metr-la_DGCRN_20260306_142338 | DGCRN | metr-la | 0.3077 | 0.6284 | 124.19 |
| metr-la_STICformer_20260306_142338 | STICformer | metr-la | 0.3096 | 0.6089 | 113.71 |
| metr-la_PatchSTG_20260306_142338 | PatchSTG | metr-la | 0.3082 | 0.6086 | 111.69 |

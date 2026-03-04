# Relatório de Comparação de Experimentos GNN

**Gerado em:** 2026-03-04 01:50:11

**Total de experimentos:** 12

## Resumo por Modelo

### DCRNN

- Experimentos: 2
- MAE médio: 0.3062 ± 0.0226
- RMSE médio: 0.5796 ± 0.0629
- MAPE médio: 147.31% ± 32.32%

### GraphWaveNet

- Experimentos: 2
- MAE médio: 0.2652 ± 0.0681
- RMSE médio: 0.5363 ± 0.1172
- MAPE médio: 121.28% ± 13.30%

### MTGNN

- Experimentos: 2
- MAE médio: 0.2898 ± 0.0757
- RMSE médio: 0.5725 ± 0.0879
- MAPE médio: 128.87% ± 7.13%

### DGCRN

- Experimentos: 2
- MAE médio: 0.2612 ± 0.0397
- RMSE médio: 0.5535 ± 0.1073
- MAPE médio: 134.23% ± 15.86%

### STICformer

- Experimentos: 2
- MAE médio: 0.2468 ± 0.0636
- RMSE médio: 0.5256 ± 0.1233
- MAPE médio: 114.85% ± 5.41%

### PatchSTG

- Experimentos: 2
- MAE médio: 0.2580 ± 0.0715
- RMSE médio: 0.5240 ± 0.1270
- MAPE médio: 119.05% ± 5.42%

## Melhores Resultados por Dataset

### pems-bay

- **Melhor modelo:** STICformer
- **Experimento:** pems-bay_STICformer_20260303_202038
- **MAE:** 0.2018
- **RMSE:** 0.4384
- **MAPE:** 118.68%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'num_layers': 2, 'num_heads': 4, 'ff_multiplier': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 30, 'patience': 5}`

### metr-la

- **Melhor modelo:** DGCRN
- **Experimento:** metr-la_DGCRN_20260303_202038
- **MAE:** 0.2892
- **RMSE:** 0.6294
- **MAPE:** 123.01%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'node_dim': 16, 'gcn_depth': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 30, 'patience': 5}`

## Tabela Completa de Resultados

| Experimento | Modelo | Dataset | MAE | RMSE | MAPE (%) |
|-------------|--------|---------|-----|------|----------|
| pems-bay_DCRNN_20260303_202038 | DCRNN | pems-bay | 0.2902 | 0.5351 | 170.16 |
| pems-bay_GraphWaveNet_20260303_202038 | GraphWaveNet | pems-bay | 0.2170 | 0.4534 | 130.69 |
| pems-bay_MTGNN_20260303_202038 | MTGNN | pems-bay | 0.2363 | 0.5104 | 133.91 |
| pems-bay_DGCRN_20260303_202038 | DGCRN | pems-bay | 0.2331 | 0.4776 | 145.44 |
| pems-bay_STICformer_20260303_202038 | STICformer | pems-bay | 0.2018 | 0.4384 | 118.68 |
| pems-bay_PatchSTG_20260303_202038 | PatchSTG | pems-bay | 0.2074 | 0.4342 | 122.89 |
| metr-la_DCRNN_20260303_202038 | DCRNN | metr-la | 0.3222 | 0.6241 | 124.45 |
| metr-la_GraphWaveNet_20260303_202038 | GraphWaveNet | metr-la | 0.3133 | 0.6192 | 111.88 |
| metr-la_MTGNN_20260303_202038 | MTGNN | metr-la | 0.3433 | 0.6347 | 123.83 |
| metr-la_DGCRN_20260303_202038 | DGCRN | metr-la | 0.2892 | 0.6294 | 123.01 |
| metr-la_STICformer_20260303_202038 | STICformer | metr-la | 0.2918 | 0.6128 | 111.02 |
| metr-la_PatchSTG_20260303_202038 | PatchSTG | metr-la | 0.3085 | 0.6138 | 115.22 |

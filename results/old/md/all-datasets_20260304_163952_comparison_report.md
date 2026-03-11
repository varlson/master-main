# Relatório de Comparação de Experimentos GNN

**Gerado em:** 2026-03-04 23:15:31

**Total de experimentos:** 12

## Resumo por Modelo

### DCRNN

- Experimentos: 2
- MAE médio: 0.3143 ± 0.0427
- RMSE médio: 0.5776 ± 0.0670
- MAPE médio: 144.61% ± 33.10%

### GraphWaveNet

- Experimentos: 2
- MAE médio: 0.2691 ± 0.0633
- RMSE médio: 0.5375 ± 0.1166
- MAPE médio: 120.70% ± 13.83%

### MTGNN

- Experimentos: 2
- MAE médio: 0.2914 ± 0.0785
- RMSE médio: 0.5735 ± 0.0855
- MAPE médio: 126.56% ± 5.96%

### DGCRN

- Experimentos: 2
- MAE médio: 0.2726 ± 0.0043
- RMSE médio: 0.5692 ± 0.0874
- MAPE médio: 157.32% ± 44.83%

### STICformer

- Experimentos: 2
- MAE médio: 0.2531 ± 0.0731
- RMSE médio: 0.5231 ± 0.1273
- MAPE médio: 118.64% ± 1.11%

### PatchSTG

- Experimentos: 2
- MAE médio: 0.2510 ± 0.0604
- RMSE médio: 0.5220 ± 0.1237
- MAPE médio: 120.57% ± 6.13%

## Melhores Resultados por Dataset

### pems-bay

- **Melhor modelo:** STICformer
- **Experimento:** pems-bay_STICformer_20260304_163952
- **MAE:** 0.2014
- **RMSE:** 0.4331
- **MAPE:** 117.86%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'num_layers': 2, 'num_heads': 4, 'ff_multiplier': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 50, 'patience': 5}`

### metr-la

- **Melhor modelo:** DGCRN
- **Experimento:** metr-la_DGCRN_20260304_163952
- **MAE:** 0.2757
- **RMSE:** 0.6311
- **MAPE:** 125.62%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'node_dim': 16, 'gcn_depth': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 50, 'patience': 5}`

## Tabela Completa de Resultados

| Experimento | Modelo | Dataset | MAE | RMSE | MAPE (%) |
|-------------|--------|---------|-----|------|----------|
| pems-bay_DCRNN_20260304_163952 | DCRNN | pems-bay | 0.2841 | 0.5303 | 168.02 |
| pems-bay_GraphWaveNet_20260304_163952 | GraphWaveNet | pems-bay | 0.2244 | 0.4550 | 130.48 |
| pems-bay_MTGNN_20260304_163952 | MTGNN | pems-bay | 0.2359 | 0.5131 | 130.77 |
| pems-bay_DGCRN_20260304_163952 | DGCRN | pems-bay | 0.2696 | 0.5074 | 189.02 |
| pems-bay_STICformer_20260304_163952 | STICformer | pems-bay | 0.2014 | 0.4331 | 117.86 |
| pems-bay_PatchSTG_20260304_163952 | PatchSTG | pems-bay | 0.2083 | 0.4345 | 124.90 |
| metr-la_DCRNN_20260304_163952 | DCRNN | metr-la | 0.3445 | 0.6250 | 121.20 |
| metr-la_GraphWaveNet_20260304_163952 | GraphWaveNet | metr-la | 0.3139 | 0.6200 | 110.92 |
| metr-la_MTGNN_20260304_163952 | MTGNN | metr-la | 0.3468 | 0.6339 | 122.35 |
| metr-la_DGCRN_20260304_163952 | DGCRN | metr-la | 0.2757 | 0.6311 | 125.62 |
| metr-la_STICformer_20260304_163952 | STICformer | metr-la | 0.3048 | 0.6131 | 119.42 |
| metr-la_PatchSTG_20260304_163952 | PatchSTG | metr-la | 0.2937 | 0.6095 | 116.23 |

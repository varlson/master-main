# Relatório de Comparação de Experimentos GNN

**Gerado em:** 2026-03-04 23:15:30

**Total de experimentos:** 6

## Resumo por Modelo

### DCRNN

- Experimentos: 1
- MAE médio: 0.3445 ± nan
- RMSE médio: 0.6250 ± nan
- MAPE médio: 121.20% ± nan%

### GraphWaveNet

- Experimentos: 1
- MAE médio: 0.3139 ± nan
- RMSE médio: 0.6200 ± nan
- MAPE médio: 110.92% ± nan%

### MTGNN

- Experimentos: 1
- MAE médio: 0.3468 ± nan
- RMSE médio: 0.6339 ± nan
- MAPE médio: 122.35% ± nan%

### DGCRN

- Experimentos: 1
- MAE médio: 0.2757 ± nan
- RMSE médio: 0.6311 ± nan
- MAPE médio: 125.62% ± nan%

### STICformer

- Experimentos: 1
- MAE médio: 0.3048 ± nan
- RMSE médio: 0.6131 ± nan
- MAPE médio: 119.42% ± nan%

### PatchSTG

- Experimentos: 1
- MAE médio: 0.2937 ± nan
- RMSE médio: 0.6095 ± nan
- MAPE médio: 116.23% ± nan%

## Melhores Resultados por Dataset

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
| metr-la_DCRNN_20260304_163952 | DCRNN | metr-la | 0.3445 | 0.6250 | 121.20 |
| metr-la_GraphWaveNet_20260304_163952 | GraphWaveNet | metr-la | 0.3139 | 0.6200 | 110.92 |
| metr-la_MTGNN_20260304_163952 | MTGNN | metr-la | 0.3468 | 0.6339 | 122.35 |
| metr-la_DGCRN_20260304_163952 | DGCRN | metr-la | 0.2757 | 0.6311 | 125.62 |
| metr-la_STICformer_20260304_163952 | STICformer | metr-la | 0.3048 | 0.6131 | 119.42 |
| metr-la_PatchSTG_20260304_163952 | PatchSTG | metr-la | 0.2937 | 0.6095 | 116.23 |

# Relatório de Comparação de Experimentos GNN

**Gerado em:** 2026-03-04 20:59:02

**Total de experimentos:** 6

## Resumo por Modelo

### DCRNN

- Experimentos: 1
- MAE médio: 0.2841 ± nan
- RMSE médio: 0.5303 ± nan
- MAPE médio: 168.02% ± nan%

### GraphWaveNet

- Experimentos: 1
- MAE médio: 0.2244 ± nan
- RMSE médio: 0.4550 ± nan
- MAPE médio: 130.48% ± nan%

### MTGNN

- Experimentos: 1
- MAE médio: 0.2359 ± nan
- RMSE médio: 0.5131 ± nan
- MAPE médio: 130.77% ± nan%

### DGCRN

- Experimentos: 1
- MAE médio: 0.2696 ± nan
- RMSE médio: 0.5074 ± nan
- MAPE médio: 189.02% ± nan%

### STICformer

- Experimentos: 1
- MAE médio: 0.2014 ± nan
- RMSE médio: 0.4331 ± nan
- MAPE médio: 117.86% ± nan%

### PatchSTG

- Experimentos: 1
- MAE médio: 0.2083 ± nan
- RMSE médio: 0.4345 ± nan
- MAPE médio: 124.90% ± nan%

## Melhores Resultados por Dataset

### pems-bay

- **Melhor modelo:** STICformer
- **Experimento:** pems-bay_STICformer_20260304_163952
- **MAE:** 0.2014
- **RMSE:** 0.4331
- **MAPE:** 117.86%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'num_layers': 2, 'num_heads': 4, 'ff_multiplier': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 50, 'patience': 5}`

## Tabela Completa de Resultados

| Experimento | Modelo | Dataset | MAE | RMSE | MAPE (%) |
|-------------|--------|---------|-----|------|----------|
| pems-bay_DCRNN_20260304_163952 | DCRNN | pems-bay | 0.2841 | 0.5303 | 168.02 |
| pems-bay_GraphWaveNet_20260304_163952 | GraphWaveNet | pems-bay | 0.2244 | 0.4550 | 130.48 |
| pems-bay_MTGNN_20260304_163952 | MTGNN | pems-bay | 0.2359 | 0.5131 | 130.77 |
| pems-bay_DGCRN_20260304_163952 | DGCRN | pems-bay | 0.2696 | 0.5074 | 189.02 |
| pems-bay_STICformer_20260304_163952 | STICformer | pems-bay | 0.2014 | 0.4331 | 117.86 |
| pems-bay_PatchSTG_20260304_163952 | PatchSTG | pems-bay | 0.2083 | 0.4345 | 124.90 |

# Relatório de Comparação de Experimentos GNN

**Gerado em:** 2026-03-03 23:39:39

**Total de experimentos:** 6

## Resumo por Modelo

### DCRNN

- Experimentos: 1
- MAE médio: 0.2902 ± nan
- RMSE médio: 0.5351 ± nan
- MAPE médio: 170.16% ± nan%

### GraphWaveNet

- Experimentos: 1
- MAE médio: 0.2170 ± nan
- RMSE médio: 0.4534 ± nan
- MAPE médio: 130.69% ± nan%

### MTGNN

- Experimentos: 1
- MAE médio: 0.2363 ± nan
- RMSE médio: 0.5104 ± nan
- MAPE médio: 133.91% ± nan%

### DGCRN

- Experimentos: 1
- MAE médio: 0.2331 ± nan
- RMSE médio: 0.4776 ± nan
- MAPE médio: 145.44% ± nan%

### STICformer

- Experimentos: 1
- MAE médio: 0.2018 ± nan
- RMSE médio: 0.4384 ± nan
- MAPE médio: 118.68% ± nan%

### PatchSTG

- Experimentos: 1
- MAE médio: 0.2074 ± nan
- RMSE médio: 0.4342 ± nan
- MAPE médio: 122.89% ± nan%

## Melhores Resultados por Dataset

### pems-bay

- **Melhor modelo:** STICformer
- **Experimento:** pems-bay_STICformer_20260303_202038
- **MAE:** 0.2018
- **RMSE:** 0.4384
- **MAPE:** 118.68%
- **Parâmetros:** `{'input_dim': 1, 'hidden_dim': 64, 'output_dim': 1, 'seq_len': 12, 'horizon': 12, 'num_layers': 2, 'num_heads': 4, 'ff_multiplier': 2, 'dropout': 0.1, 'lr': 0.001, 'weight_decay': 0.0001, 'epochs': 30, 'patience': 5}`

## Tabela Completa de Resultados

| Experimento | Modelo | Dataset | MAE | RMSE | MAPE (%) |
|-------------|--------|---------|-----|------|----------|
| pems-bay_DCRNN_20260303_202038 | DCRNN | pems-bay | 0.2902 | 0.5351 | 170.16 |
| pems-bay_GraphWaveNet_20260303_202038 | GraphWaveNet | pems-bay | 0.2170 | 0.4534 | 130.69 |
| pems-bay_MTGNN_20260303_202038 | MTGNN | pems-bay | 0.2363 | 0.5104 | 133.91 |
| pems-bay_DGCRN_20260303_202038 | DGCRN | pems-bay | 0.2331 | 0.4776 | 145.44 |
| pems-bay_STICformer_20260303_202038 | STICformer | pems-bay | 0.2018 | 0.4384 | 118.68 |
| pems-bay_PatchSTG_20260303_202038 | PatchSTG | pems-bay | 0.2074 | 0.4342 | 122.89 |

# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 19:40:52

**Total de experimentos consolidados:** 6

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9972 ± 0.0000 | 11.8574 ± 0.0000 | 11.82 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 6.1061 ± 0.0000 | 12.0270 ± 0.0000 | 12.03 ± 0.00 | 1.82 |
| 3 | MTGNN | 6.8111 ± 0.0000 | 12.3707 ± 0.0000 | 13.42 ± 0.00 | 13.57 |

### pems-bay

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 1.9505 ± 0.0000 | 4.1133 ± 0.0000 | 3.12 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 2.0388 ± 0.0000 | 4.2918 ± 0.0000 | 3.26 ± 0.00 | 4.53 |
| 3 | MTGNN | 2.1971 ± 0.0000 | 4.7311 ± 0.0000 | 3.52 ± 0.00 | 12.64 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 2 | 1.00 | 3.9738 | 7.9853 | 7.47 |
| GraphWaveNet | 2 | 2.00 | 4.0724 | 8.1594 | 7.65 |
| MTGNN | 2 | 3.00 | 4.5041 | 8.5509 | 8.47 |

## Configurações Selecionadas

### simple_metr-la_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.1875
- Teste (MAE): 5.9972
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_metr-la_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.3883
- Teste (MAE): 6.1061
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_metr-la_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: metr-la
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.8495
- Teste (MAE): 6.8111
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_pems-bay_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: pems-bay
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.0737
- Teste (MAE): 1.9505
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_pems-bay_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: pems-bay
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.1728
- Teste (MAE): 2.0388
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_pems-bay_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: pems-bay
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3528
- Teste (MAE): 2.1971
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_metr-la_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | metr-la | 1 | 5.9972 | 11.8574 | 34.23 | 11.82 | 1 |
| simple_metr-la_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | metr-la | 1 | 6.1061 | 12.0270 | 34.65 | 12.03 | 2 |
| simple_metr-la_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | metr-la | 1 | 6.8111 | 12.3707 | 35.20 | 13.42 | 3 |
| simple_pems-bay_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | pems-bay | 1 | 1.9505 | 4.1133 | 3.92 | 3.12 | 1 |
| simple_pems-bay_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | pems-bay | 1 | 2.0388 | 4.2918 | 4.12 | 3.26 | 2 |
| simple_pems-bay_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | pems-bay | 1 | 2.1971 | 4.7311 | 4.47 | 3.52 | 3 |

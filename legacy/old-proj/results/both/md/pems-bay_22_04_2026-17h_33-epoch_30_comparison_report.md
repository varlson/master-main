# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 19:15:30

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### pems-bay

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 1.9505 ± 0.0000 | 4.1133 ± 0.0000 | 3.12 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 2.0388 ± 0.0000 | 4.2918 ± 0.0000 | 3.26 ± 0.00 | 4.53 |
| 3 | MTGNN | 2.1971 ± 0.0000 | 4.7311 ± 0.0000 | 3.52 ± 0.00 | 12.64 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 1.9505 | 4.1133 | 3.12 |
| GraphWaveNet | 1 | 2.00 | 2.0388 | 4.2918 | 3.26 |
| MTGNN | 1 | 3.00 | 2.1971 | 4.7311 | 3.52 |

## Configurações Selecionadas

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
| simple_pems-bay_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | pems-bay | 1 | 1.9505 | 4.1133 | 3.92 | 3.12 | 1 |
| simple_pems-bay_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | pems-bay | 1 | 2.0388 | 4.2918 | 4.12 | 3.26 | 2 |
| simple_pems-bay_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | pems-bay | 1 | 2.1971 | 4.7311 | 4.47 | 3.52 | 3 |

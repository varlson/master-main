# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 12:47:49

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### pems-bay-by-nois_corr-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 2.1643 ± 0.0000 | 4.4357 ± 0.0000 | 3.47 ± 0.00 | 0.00 |
| 2 | MTGNN | 2.2685 ± 0.0000 | 4.8414 ± 0.0000 | 3.63 ± 0.00 | 4.81 |
| 3 | GraphWaveNet | 2.5557 ± 0.0000 | 5.1512 ± 0.0000 | 4.09 ± 0.00 | 18.08 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 2.1643 | 4.4357 | 3.47 |
| MTGNN | 1 | 2.00 | 2.2685 | 4.8414 | 3.63 |
| GraphWaveNet | 1 | 3.00 | 2.5557 | 5.1512 | 4.09 |

## Configurações Selecionadas

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3343
- Teste (MAE): 2.1643
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.4856
- Teste (MAE): 2.2685
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.7675
- Teste (MAE): 2.5557
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.1643 | 4.4357 | 4.32 | 3.47 | 1 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.2685 | 4.8414 | 4.60 | 3.63 | 2 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.5557 | 5.1512 | 5.36 | 4.09 | 3 |

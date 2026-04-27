# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 20:23:28

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la-by-nois_corr-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9493 ± 0.0000 | 11.8476 ± 0.0000 | 11.73 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 6.1305 ± 0.0000 | 12.0339 ± 0.0000 | 12.08 ± 0.00 | 3.05 |
| 3 | MTGNN | 6.7817 ± 0.0000 | 12.3646 ± 0.0000 | 13.37 ± 0.00 | 13.99 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 5.9493 | 11.8476 | 11.73 |
| GraphWaveNet | 1 | 2.00 | 6.1305 | 12.0339 | 12.08 |
| MTGNN | 1 | 3.00 | 6.7817 | 12.3646 | 13.37 |

## Configurações Selecionadas

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.1292
- Teste (MAE): 5.9493
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.3406
- Teste (MAE): 6.1305
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.8245
- Teste (MAE): 6.7817
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 5.9493 | 11.8476 | 34.03 | 11.73 | 1 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 6.1305 | 12.0339 | 34.41 | 12.08 | 2 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 6.7817 | 12.3646 | 35.14 | 13.37 | 3 |

# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 20:45:11

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la-by-high_sal-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 6.0864 ± 0.0000 | 11.9126 ± 0.0000 | 11.96 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 6.2041 ± 0.0000 | 12.0241 ± 0.0000 | 12.20 ± 0.00 | 1.93 |
| 3 | MTGNN | 7.0226 ± 0.0000 | 12.3051 ± 0.0000 | 13.80 ± 0.00 | 15.38 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 6.0864 | 11.9126 | 11.96 |
| GraphWaveNet | 1 | 2.00 | 6.2041 | 12.0241 | 12.20 |
| MTGNN | 1 | 3.00 | 7.0226 | 12.3051 | 13.80 |

## Configurações Selecionadas

### simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: metr-la-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.1382
- Teste (MAE): 6.0864
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: metr-la-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.4267
- Teste (MAE): 6.2041
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: metr-la-by-high_sal-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 6.0663
- Teste (MAE): 7.0226
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 6.0864 | 11.9126 | 33.65 | 11.96 | 1 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 6.2041 | 12.0241 | 34.58 | 12.20 | 2 |
| simple_backbone_metr-la-by-high_sal-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | metr-la-by-high_sal-with-alpah_filter0_4 | 1 | 7.0226 | 12.3051 | 35.29 | 13.80 | 3 |

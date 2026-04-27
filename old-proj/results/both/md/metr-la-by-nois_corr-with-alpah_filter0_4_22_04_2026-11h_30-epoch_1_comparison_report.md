# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 11:53:21

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la-by-nois_corr-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9981 ± 0.0000 | 12.0306 ± 0.0000 | 11.82 ± 0.00 | 0.00 |
| 2 | MTGNN | 6.8843 ± 0.0000 | 12.3703 ± 0.0000 | 13.57 ± 0.00 | 14.78 |
| 3 | GraphWaveNet | 7.4615 ± 0.0000 | 12.4822 ± 0.0000 | 14.71 ± 0.00 | 24.40 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 5.9981 | 12.0306 | 11.82 |
| MTGNN | 1 | 2.00 | 6.8843 | 12.3703 | 13.57 |
| GraphWaveNet | 1 | 3.00 | 7.4615 | 12.4822 | 14.71 |

## Configurações Selecionadas

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.3035
- Teste (MAE): 5.9981
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.9360
- Teste (MAE): 6.8843
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: metr-la-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 6.5518
- Teste (MAE): 7.4615
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 5.9981 | 12.0306 | 34.59 | 11.82 | 1 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 6.8843 | 12.3703 | 35.27 | 13.57 | 2 |
| simple_backbone_metr-la-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | metr-la-by-nois_corr-with-alpah_filter0_4 | 1 | 7.4615 | 12.4822 | 35.74 | 14.71 | 3 |

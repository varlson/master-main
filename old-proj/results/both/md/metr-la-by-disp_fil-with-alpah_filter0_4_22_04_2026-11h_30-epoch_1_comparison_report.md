# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 11:42:09

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la-by-disp_fil-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9449 ± 0.0000 | 12.0806 ± 0.0000 | 11.72 ± 0.00 | 0.00 |
| 2 | MTGNN | 6.8223 ± 0.0000 | 12.3648 ± 0.0000 | 13.45 ± 0.00 | 14.76 |
| 3 | GraphWaveNet | 7.0205 ± 0.0000 | 12.3617 ± 0.0000 | 13.84 ± 0.00 | 18.09 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 5.9449 | 12.0806 | 11.72 |
| MTGNN | 1 | 2.00 | 6.8223 | 12.3648 | 13.45 |
| GraphWaveNet | 1 | 3.00 | 7.0205 | 12.3617 | 13.84 |

## Configurações Selecionadas

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.2785
- Teste (MAE): 5.9449
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.8859
- Teste (MAE): 6.8223
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 6.1700
- Teste (MAE): 7.0205
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 5.9449 | 12.0806 | 34.67 | 11.72 | 1 |
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 6.8223 | 12.3648 | 35.24 | 13.45 | 2 |
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 7.0205 | 12.3617 | 35.41 | 13.84 | 3 |

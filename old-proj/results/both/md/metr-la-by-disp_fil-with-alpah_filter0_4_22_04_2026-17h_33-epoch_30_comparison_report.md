# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 20:00:29

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### metr-la-by-disp_fil-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 5.9656 ± 0.0000 | 11.8588 ± 0.0000 | 11.76 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 6.3341 ± 0.0000 | 12.0910 ± 0.0000 | 12.48 ± 0.00 | 6.18 |
| 3 | MTGNN | 6.8253 ± 0.0000 | 12.3840 ± 0.0000 | 13.45 ± 0.00 | 14.41 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 5.9656 | 11.8588 | 11.76 |
| GraphWaveNet | 1 | 2.00 | 6.3341 | 12.0910 | 12.48 |
| MTGNN | 1 | 3.00 | 6.8253 | 12.3840 | 13.45 |

## Configurações Selecionadas

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.1705
- Teste (MAE): 5.9656
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.4720
- Teste (MAE): 6.3341
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: metr-la-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 5.8570
- Teste (MAE): 6.8253
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 5.9656 | 11.8588 | 34.12 | 11.76 | 1 |
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 6.3341 | 12.0910 | 34.46 | 12.48 | 2 |
| simple_backbone_metr-la-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | metr-la-by-disp_fil-with-alpah_filter0_4 | 1 | 6.8253 | 12.3840 | 35.16 | 13.45 | 3 |

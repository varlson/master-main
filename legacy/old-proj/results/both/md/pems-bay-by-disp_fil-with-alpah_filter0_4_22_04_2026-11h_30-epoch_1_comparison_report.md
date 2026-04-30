# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 12:25:44

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### pems-bay-by-disp_fil-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 2.1766 ± 0.0000 | 4.4707 ± 0.0000 | 3.48 ± 0.00 | 0.00 |
| 2 | MTGNN | 2.2750 ± 0.0000 | 4.8442 ± 0.0000 | 3.64 ± 0.00 | 4.52 |
| 3 | GraphWaveNet | 2.4358 ± 0.0000 | 5.0032 ± 0.0000 | 3.90 ± 0.00 | 11.91 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 2.1766 | 4.4707 | 3.48 |
| MTGNN | 1 | 2.00 | 2.2750 | 4.8442 | 3.64 |
| GraphWaveNet | 1 | 3.00 | 2.4358 | 5.0032 | 3.90 |

## Configurações Selecionadas

### simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1

- Modelo: STICformer
- Dataset: pems-bay-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3483
- Teste (MAE): 2.1766
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1

- Modelo: MTGNN
- Dataset: pems-bay-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.4933
- Teste (MAE): 2.2750
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1

- Modelo: GraphWaveNet
- Dataset: pems-bay-by-disp_fil-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.6449
- Teste (MAE): 2.4358
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_STICformer_22_04_2026-11h_30-epoch_1 | STICformer | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.1766 | 4.4707 | 4.34 | 3.48 | 1 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_MTGNN_22_04_2026-11h_30-epoch_1 | MTGNN | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.2750 | 4.8442 | 4.61 | 3.64 | 2 |
| simple_backbone_pems-bay-by-disp_fil-with-alpah_filter0_4_GraphWaveNet_22_04_2026-11h_30-epoch_1 | GraphWaveNet | pems-bay-by-disp_fil-with-alpah_filter0_4 | 1 | 2.4358 | 5.0032 | 5.02 | 3.90 | 3 |

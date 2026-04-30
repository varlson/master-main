# Relatório de Comparação Científica de Experimentos

**Gerado em:** 2026-04-22 23:15:06

**Total de experimentos consolidados:** 3

Os números abaixo representam desempenho final em teste para a configuração selecionada por validação, agregada por múltiplas seeds quando disponíveis.

## Ranking por Dataset

### pems-bay-by-nois_corr-with-alpah_filter0_4

| Rank | Modelo | Test MAE | Test RMSE | Test WAPE (%) | Delta vs melhor (%) |
|------|--------|----------|-----------|---------------|----------------------|
| 1 | STICformer | 1.8877 ± 0.0000 | 4.0166 ± 0.0000 | 3.02 ± 0.00 | 0.00 |
| 2 | GraphWaveNet | 2.0912 ± 0.0000 | 4.3888 ± 0.0000 | 3.35 ± 0.00 | 10.78 |
| 3 | MTGNN | 2.1868 ± 0.0000 | 4.6290 ± 0.0000 | 3.50 ± 0.00 | 15.84 |

## Resumo por Modelo

| Modelo | Datasets | Rank médio | MAE médio | RMSE médio | WAPE médio (%) |
|--------|----------|------------|-----------|------------|----------------|
| STICformer | 1 | 1.00 | 1.8877 | 4.0166 | 3.02 |
| GraphWaveNet | 1 | 2.00 | 2.0912 | 4.3888 | 3.35 |
| MTGNN | 1 | 3.00 | 2.1868 | 4.6290 | 3.50 |

## Configurações Selecionadas

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30

- Modelo: STICformer
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 1.9917
- Teste (MAE): 1.8877
- Parâmetros: `{"dropout": 0.1, "ff_multiplier": 2, "hidden_dim": 64, "lr": 0.001, "num_heads": 4, "num_layers": 2, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30

- Modelo: GraphWaveNet
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.2679
- Teste (MAE): 2.0912
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "hidden_dim": 64, "k": 2, "lr": 0.001, "num_blocks": 3, "patience": 5, "weight_decay": 0.0001}`

### simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30

- Modelo: MTGNN
- Dataset: pems-bay-by-nois_corr-with-alpah_filter0_4
- Métrica de seleção: `val_mae`
- Seeds finais: 1
- Melhor validação (MAE): 2.3355
- Teste (MAE): 2.1868
- Parâmetros: `{"dilation_base": 2, "dropout": 0.1, "gcn_depth": 2, "hidden_dim": 64, "kernel_size": 2, "lr": 0.001, "node_dim": 16, "num_blocks": 3, "patience": 5, "propalpha": 0.05, "weight_decay": 0.0001}`

## Tabela Completa

| Experimento | Modelo | Dataset | Seeds | Test MAE | Test RMSE | Test sMAPE (%) | Test WAPE (%) | Rank |
|-------------|--------|---------|-------|----------|-----------|----------------|----------------|------|
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_STICformer_22_04_2026-17h_33-epoch_30 | STICformer | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 1.8877 | 4.0166 | 3.79 | 3.02 | 1 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_GraphWaveNet_22_04_2026-17h_33-epoch_30 | GraphWaveNet | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.0912 | 4.3888 | 4.22 | 3.35 | 2 |
| simple_backbone_pems-bay-by-nois_corr-with-alpah_filter0_4_MTGNN_22_04_2026-17h_33-epoch_30 | MTGNN | pems-bay-by-nois_corr-with-alpah_filter0_4 | 1 | 2.1868 | 4.6290 | 4.42 | 3.50 | 3 |

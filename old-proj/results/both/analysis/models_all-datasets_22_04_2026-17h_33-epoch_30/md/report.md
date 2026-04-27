# Model Analysis Pipeline

Gerado em: 2026-04-24 10:59:54

- Modo: `both`
- Metrica principal: `MAE`
- Alpha: `0.05`
- Normalizacao radar: `relative_score`

## Fontes

- `backbone` md: `/home/suleimane/master/main/master-main/results/both/md/backbone_all-datasets_22_04_2026-17h_33-epoch_30_comparison_report.md`
- `backbone` csv: `/home/suleimane/master/main/master-main/results/both/csv/backbone_all-datasets_22_04_2026-17h_33-epoch_30_consolidated_experiments.csv`
- `original` md: `/home/suleimane/master/main/master-main/results/both/md/original_all-datasets_22_04_2026-17h_33-epoch_30_comparison_report.md`
- `original` csv: `/home/suleimane/master/main/master-main/results/both/csv/original_all-datasets_22_04_2026-17h_33-epoch_30_consolidated_experiments.csv`

## Tabelas

### Comparacoes Original vs Backbone por dataset

- `METR-LA` md: `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/tables/metr-la.md`
- `METR-LA` latex: `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/tables/metr-la.latex`
- `PEMS-BAY` md: `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/tables/pems-bay.md`
- `PEMS-BAY` latex: `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/tables/pems-bay.latex`

## Radar Charts

### models_full_by_dataset

- `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/radar/models_full/radar_models_metrla_full.png`
- `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/radar/models_full/radar_models_pemsbay_full.png`

### backbone_by_model

- `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/radar/backbone_by_model/radar_backbone_GraphWaveNet.png`
- `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/radar/backbone_by_model/radar_backbone_MTGNN.png`
- `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/radar/backbone_by_model/radar_backbone_STICformer.png`

### backbones_by_dataset

- `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/radar/backbones_by_dataset/radar_backbones_metrla.png`
- `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/radar/backbones_by_dataset/radar_backbones_pemsbay.png`

## Testes Estatisticos

### backbones_general

- unit_col: `backbone`
- block_cols: `['dataset_label', 'model']`

#### MAE

- Friedman: estatistica = `5.2000`, p-valor = `0.157724`
- Critical Difference: `1.9148`
- Average ranks:
  - `NC`: `1.6667`
  - `Full`: `2.3333`
  - `HSS`: `2.6667`
  - `DF`: `3.3333`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/ranks_backbones_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/nemenyi_backbones_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/pvalues_backbones_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/nemenyi_backbones_MAE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_general/cd_backbones_MAE.png`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_general/cd_backbones_MAE.pdf`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/stats_backbones_MAE.json`

#### RMSE

- Friedman: estatistica = `9.0000`, p-valor = `0.029291`
- Critical Difference: `1.9148`
- Average ranks:
  - `NC`: `1.8333`
  - `HSS`: `2.0000`
  - `Full`: `2.3333`
  - `DF`: `3.8333`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/ranks_backbones_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/nemenyi_backbones_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/pvalues_backbones_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/nemenyi_backbones_RMSE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/stats_backbones_RMSE.json`

#### WAPE

- Friedman: estatistica = `5.2000`, p-valor = `0.157724`
- Critical Difference: `1.9148`
- Average ranks:
  - `NC`: `1.6667`
  - `Full`: `2.3333`
  - `HSS`: `2.6667`
  - `DF`: `3.3333`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/ranks_backbones_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/nemenyi_backbones_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/pvalues_backbones_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/nemenyi_backbones_WAPE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_general/stats_backbones_WAPE.json`

### models_general

- unit_col: `model`
- block_cols: `['dataset_label', 'backbone']`

#### MAE

- Friedman: estatistica = `16.0000`, p-valor = `0.000335`
- Critical Difference: `1.1719`
- Average ranks:
  - `STICformer`: `1.0000`
  - `GraphWaveNet`: `2.0000`
  - `MTGNN`: `3.0000`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/ranks_models_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/nemenyi_models_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/pvalues_models_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/nemenyi_models_MAE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/stats_models_MAE.json`

#### RMSE

- Friedman: estatistica = `16.0000`, p-valor = `0.000335`
- Critical Difference: `1.1719`
- Average ranks:
  - `STICformer`: `1.0000`
  - `GraphWaveNet`: `2.0000`
  - `MTGNN`: `3.0000`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/ranks_models_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/nemenyi_models_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/pvalues_models_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/nemenyi_models_RMSE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/stats_models_RMSE.json`

#### WAPE

- Friedman: estatistica = `16.0000`, p-valor = `0.000335`
- Critical Difference: `1.1719`
- Average ranks:
  - `STICformer`: `1.0000`
  - `GraphWaveNet`: `2.0000`
  - `MTGNN`: `3.0000`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/ranks_models_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/nemenyi_models_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/pvalues_models_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/nemenyi_models_WAPE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/models_general/stats_models_WAPE.json`

### backbones_by_dataset

### METR-LA

- unit_col: `backbone`
- block_cols: `['model']`

#### MAE

- Friedman: estatistica = `5.8000`, p-valor = `0.121757`
- Critical Difference: `2.7080`
- Average ranks:
  - `NC`: `1.3333`
  - `Full`: `2.0000`
  - `DF`: `3.0000`
  - `HSS`: `3.6667`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/ranks_backbones_metrla_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/nemenyi_backbones_metrla_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/pvalues_backbones_metrla_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/nemenyi_backbones_metrla_MAE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/metrla/cd_backbones_metrla_MAE.png`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/metrla/cd_backbones_metrla_MAE.pdf`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/stats_backbones_metrla_MAE.json`

#### RMSE

- Friedman: estatistica = `3.4000`, p-valor = `0.333965`
- Critical Difference: `2.7080`
- Average ranks:
  - `NC`: `2.0000`
  - `HSS`: `2.0000`
  - `Full`: `2.3333`
  - `DF`: `3.6667`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/ranks_backbones_metrla_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/nemenyi_backbones_metrla_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/pvalues_backbones_metrla_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/nemenyi_backbones_metrla_RMSE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/metrla/cd_backbones_metrla_RMSE.png`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/metrla/cd_backbones_metrla_RMSE.pdf`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/stats_backbones_metrla_RMSE.json`

#### WAPE

- Friedman: estatistica = `5.8000`, p-valor = `0.121757`
- Critical Difference: `2.7080`
- Average ranks:
  - `NC`: `1.3333`
  - `Full`: `2.0000`
  - `DF`: `3.0000`
  - `HSS`: `3.6667`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/ranks_backbones_metrla_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/nemenyi_backbones_metrla_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/pvalues_backbones_metrla_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/nemenyi_backbones_metrla_WAPE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/metrla/cd_backbones_metrla_WAPE.png`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/metrla/cd_backbones_metrla_WAPE.pdf`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/metrla/stats_backbones_metrla_WAPE.json`

### PEMS-BAY

- unit_col: `backbone`
- block_cols: `['model']`

#### MAE

- Friedman: estatistica = `4.2000`, p-valor = `0.240662`
- Critical Difference: `2.7080`
- Average ranks:
  - `HSS`: `1.6667`
  - `NC`: `2.0000`
  - `Full`: `2.6667`
  - `DF`: `3.6667`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/ranks_backbones_pemsbay_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/nemenyi_backbones_pemsbay_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/pvalues_backbones_pemsbay_MAE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/nemenyi_backbones_pemsbay_MAE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/pemsbay/cd_backbones_pemsbay_MAE.png`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/pemsbay/cd_backbones_pemsbay_MAE.pdf`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/stats_backbones_pemsbay_MAE.json`

#### RMSE

- Friedman: estatistica = `5.8000`, p-valor = `0.121757`
- Critical Difference: `2.7080`
- Average ranks:
  - `NC`: `1.6667`
  - `HSS`: `2.0000`
  - `Full`: `2.3333`
  - `DF`: `4.0000`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/ranks_backbones_pemsbay_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/nemenyi_backbones_pemsbay_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/pvalues_backbones_pemsbay_RMSE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/nemenyi_backbones_pemsbay_RMSE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/pemsbay/cd_backbones_pemsbay_RMSE.png`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/pemsbay/cd_backbones_pemsbay_RMSE.pdf`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/stats_backbones_pemsbay_RMSE.json`

#### WAPE

- Friedman: estatistica = `4.2000`, p-valor = `0.240662`
- Critical Difference: `2.7080`
- Average ranks:
  - `HSS`: `1.6667`
  - `NC`: `2.0000`
  - `Full`: `2.6667`
  - `DF`: `3.6667`
- Arquivos:
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/ranks_backbones_pemsbay_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/nemenyi_backbones_pemsbay_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/pvalues_backbones_pemsbay_WAPE.csv`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/nemenyi_backbones_pemsbay_WAPE.tex`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/pemsbay/cd_backbones_pemsbay_WAPE.png`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/figures/cd/backbones_by_dataset/pemsbay/cd_backbones_pemsbay_WAPE.pdf`
  - `/home/suleimane/master/main/master-main/results/both/analysis/models_all-datasets_22_04_2026-17h_33-epoch_30/stats/backbones_by_dataset/pemsbay/stats_backbones_pemsbay_WAPE.json`


# METR-LA

Dataset original: `metr-la`

## Original vs DF

Dataset backbone: `metr-la-by-disp_fil-with-alpah_filter0_4`

| Model | Original MAE | DF MAE | Original RMSE | DF RMSE | Original WAPE | DF WAPE | Delta MAE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STICformer | 5.997 | 5.966 | 11.857 | 11.859 | 11.820 | 11.757 | -0.53% |
| MTGNN | 6.811 | 6.825 | 12.371 | 12.384 | 13.424 | 13.452 | +0.21% |
| GraphWaveNet | 6.106 | 6.334 | 12.027 | 12.091 | 12.034 | 12.484 | +3.73% |

## Original vs NC

Dataset backbone: `metr-la-by-nois_corr-with-alpah_filter0_4`

| Model | Original MAE | NC MAE | Original RMSE | NC RMSE | Original WAPE | NC WAPE | Delta MAE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STICformer | 5.997 | 5.949 | 11.857 | 11.848 | 11.820 | 11.725 | -0.80% |
| MTGNN | 6.811 | 6.782 | 12.371 | 12.365 | 13.424 | 13.366 | -0.43% |
| GraphWaveNet | 6.106 | 6.131 | 12.027 | 12.034 | 12.034 | 12.082 | +0.40% |

## Original vs HSS

Dataset backbone: `metr-la-by-high_sal-with-alpah_filter0_4`

| Model | Original MAE | HSS MAE | Original RMSE | HSS RMSE | Original WAPE | HSS WAPE | Delta MAE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STICformer | 5.997 | 6.086 | 11.857 | 11.913 | 11.820 | 11.964 | +1.49% |
| MTGNN | 6.811 | 7.023 | 12.371 | 12.305 | 13.424 | 13.804 | +3.10% |
| GraphWaveNet | 6.106 | 6.204 | 12.027 | 12.024 | 12.034 | 12.195 | +1.60% |

## Original vs Media dos backbones

Backbones considerados: `DF, NC, HSS`

| Model | Original MAE | Backbone Mean MAE | Original RMSE | Backbone Mean RMSE | Original WAPE | Backbone Mean WAPE | Delta MAE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STICformer | 5.997 | 6.000 | 11.857 | 11.873 | 11.820 | 11.815 | +0.05% |
| MTGNN | 6.811 | 6.877 | 12.371 | 12.351 | 13.424 | 13.541 | +0.96% |
| GraphWaveNet | 6.106 | 6.223 | 12.027 | 12.050 | 12.034 | 12.254 | +1.91% |

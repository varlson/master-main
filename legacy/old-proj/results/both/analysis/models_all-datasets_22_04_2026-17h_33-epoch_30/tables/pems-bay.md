# PEMS-BAY

Dataset original: `pems-bay`

## Original vs DF

Dataset backbone: `pems-bay-by-disp_fil-with-alpah_filter0_4`

| Model | Original MAE | DF MAE | Original RMSE | DF RMSE | Original WAPE | DF WAPE | Delta MAE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STICformer | 1.950 | 1.960 | 4.113 | 4.116 | 3.123 | 3.139 | +0.51% |
| MTGNN | 2.197 | 2.194 | 4.731 | 4.740 | 3.518 | 3.512 | -0.16% |
| GraphWaveNet | 2.039 | 2.119 | 4.292 | 4.429 | 3.264 | 3.392 | +3.93% |

## Original vs NC

Dataset backbone: `pems-bay-by-nois_corr-with-alpah_filter0_4`

| Model | Original MAE | NC MAE | Original RMSE | NC RMSE | Original WAPE | NC WAPE | Delta MAE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STICformer | 1.950 | 1.888 | 4.113 | 4.017 | 3.123 | 3.022 | -3.22% |
| MTGNN | 2.197 | 2.187 | 4.731 | 4.629 | 3.518 | 3.501 | -0.47% |
| GraphWaveNet | 2.039 | 2.091 | 4.292 | 4.389 | 3.264 | 3.348 | +2.57% |

## Original vs HSS

Dataset backbone: `pems-bay-by-high_sal-with-alpah_filter0_4`

| Model | Original MAE | HSS MAE | Original RMSE | HSS RMSE | Original WAPE | HSS WAPE | Delta MAE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STICformer | 1.950 | 1.913 | 4.113 | 4.116 | 3.123 | 3.055 | -1.91% |
| MTGNN | 2.197 | 2.162 | 4.731 | 4.670 | 3.518 | 3.451 | -1.61% |
| GraphWaveNet | 2.039 | 2.069 | 4.292 | 4.239 | 3.264 | 3.304 | +1.50% |

## Original vs Media dos backbones

Backbones considerados: `DF, NC, HSS`

| Model | Original MAE | Backbone Mean MAE | Original RMSE | Backbone Mean RMSE | Original WAPE | Backbone Mean WAPE | Delta MAE (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STICformer | 1.950 | 1.920 | 4.113 | 4.083 | 3.123 | 3.072 | -1.54% |
| MTGNN | 2.197 | 2.181 | 4.731 | 4.680 | 3.518 | 3.488 | -0.75% |
| GraphWaveNet | 2.039 | 2.093 | 4.292 | 4.352 | 3.264 | 3.348 | +2.67% |

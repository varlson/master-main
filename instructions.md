ssh config file

scp -r -P 5000 suleimaneducure@200.239.132.159://media/work/suleimaneducure/codes/master-main/data /home/suleimane/master/main/master-main

Entrada principal recomendada:
python3 -m pipeline.main

Forecasting com config:
python3 -m pipeline.main forecast --config configs/search/pipeline.search.example.json
python3 -m pipeline.main forecast --config configs/run_best/pipeline.run_best.example.json
python3 -m pipeline.main forecast --config configs/full/pipeline.run_configured.example.json

Dry run:
python3 -m pipeline.main forecast --config configs/search/pipeline.search.example.json --dry-run

Backbone generation:
python3 -m pipeline.main build-backbones --datasets metr-la --methods disp_fil nois_corr --alpha 0.3

Backbone analysis:
python3 -m pipeline.main analyze-backbones --datasets metr-la pems-bay --methods disp_fil nois_corr --alpha 0.3

Backbone analysis outputs:
outputs/analysis/<run_id>/

Backbone generation outputs:
- dados derivados: data/npy e data/GraphML
- relatorios por backbone: outputs/backbone/<dataset>/<backbone_name>/

Faça analise usando os artefatos arquivados em `legacy/backbone-analisys-historical` comparando os dois metodos de extração de backbones e os dois datasets:

- como cada dataset performou com os metodos de extração de backbone
- Como os datasets combinaram ou nao com os metodos
- O que os indices calculados pós extração reveram sobre dataset originais
- e muito mais analises e observações

Obs: Suspeito de que uma certa dataset aparenta ser mais sensivel a poda e que taves essa sensibilidade tenha a ver com a estrutura topologica original e que os beneficios de extração de backbone depende da estrutura da rede original , ou seja, nem sempre backbone ajuda.

Historico arquivado:
- `legacy/backbone-analisys-historical/`

Novos relatorios:
- `outputs/analysis/`

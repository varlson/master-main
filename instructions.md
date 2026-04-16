ssh config file

Grid search:
CONFIG_SOURCE=json CONFIG_FILE=config.json python3 main.py

Selected params already saved:
CONFIG_SOURCE=json CONFIG_FILE=config.selected.example.json python3 main.py

CONFIG_SOURCE=json CONFIG_FILE=config.selected.top3.json python3 main.py

Backbone generation (alpha-only + auto analysis):
cd backbone
python3 main.py
python3 main.py --datasets metr-la --methods disp_fil nois_corr

Backbone analysis outputs:
backbone/analisys/<dataset>/<backbone_name>/

CONFIG_SOURCE=json CONFIG_FILE=config.selected.top3.json python3 main.py
python3 simple_pipeline.py --config simple_config.json --params params.json

Faça analise no /backbone/analisys comparando os dois metodos de extração de backbones e os dois datasets:

- como cada dataset performou com os metodos de extração de backbone
- Como os datasets combinaram ou nao com os metodos
- O que os indices calculados pós extração reveram sobre dataset originais
- e muito mais analises e observações

Obs: Suspeito de que uma certa dataset aparenta ser mais sensivel a poda e que taves essa sensibilidade tenha a ver com a estrutura topologica original e que os beneficios de extração de backbone depende da estrutura da rede original , ou seja, nem sempre backbone ajuda.

Gere relator.md no /backbone/analisys crie as tabelas com indices das redes originais e indices pós extração para cada dataset e metodo todo na mesma coluna. e deixe em baixo script latex para a mesma tabela que eu possa usar no documento latex depos.

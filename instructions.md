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

# Pipeline

Entrada principal unificada do projeto.

## Fluxos

- `search_best`: roda grid search apenas em datasets originais e salva as melhores configuracoes por dataset/modelo.
- `run_best`: carrega um arquivo `best_configs.json` e executa treino, previsao, relatorios e plots com os melhores parametros.
- `search_and_run`: faz busca e fase final no mesmo fluxo, ainda usando apenas datasets originais.
- `run_configured`: executa treino, validacao, teste e consolidacao usando parametros fixos definidos no proprio JSON, sem grid search.

## Como executar

```bash
python3 -m pipeline.main forecast --config configs/search/pipeline.search.example.json
python3 -m pipeline.main forecast --config configs/run_best/pipeline.run_best.example.json
python3 -m pipeline.main forecast --config configs/full/pipeline.run_configured.example.json
```

Compatibilidade:

```bash
python3 -m pipeline.main --config configs/search/pipeline.search.example.json
```

Sem subcomando explicito, o `pipeline` assume `forecast`.

## Dry run

```bash
python3 -m pipeline.main forecast --config configs/search/pipeline.search.example.json --dry-run
```

## Backbone

Geracao de backbones:

```bash
python3 -m pipeline.main build-backbones --datasets metr-la --methods disp_fil high_sal --alpha 0.3
```

Analise de backbones:

```bash
python3 -m pipeline.main analyze-backbones --datasets metr-la pems-bay --methods disp_fil nois_corr --alpha 0.3
```

Saidas:

- `build-backbones` grava os arquivos de dados derivados em `data/npy` e `data/GraphML`, e os relatorios associados em `outputs/backbone/`
- `analyze-backbones` grava tabelas, plots e markdowns em `outputs/analysis/`

## Observacoes

- A busca de hiperparametros sempre usa dados originais.
- O modo `run_configured` aceita `original`, `backbone` ou `both`, e usa `model_params` no JSON para definir os parametros de cada modelo.
- No modo `run_best`, se o dataset alvo for backbone, o pipeline tenta reaproveitar a melhor configuracao do dataset original base, por exemplo `metr-la` para `metr-la-by-high_sal-with-alpah_filter0_4`.
- O motor de treino agora prioriza os modulos ativos em `shared/` e `models/` na raiz do repositório.
- `legacy/old-proj/` permanece apenas como area legada e fonte de referencia historica, principalmente para configs e resultados antigos.
- As implementacoes de orquestracao e filtros de backbone agora vivem em `pipeline/backbone/`.
- Os scripts em `backbone/` continuam disponiveis por compatibilidade, mas o fluxo recomendado passa por `pipeline/main.py`.
- O historico antigo de artefatos de `backbone/analisys/` foi arquivado em `legacy/backbone-analisys-historical/`.

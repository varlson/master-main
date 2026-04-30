# Organizacao do Projeto

## Entrada oficial

- `python3 -m pipeline.main`

## Direcao estrutural

- `pipeline/`: pacote principal e orquestracao
- `configs/`: configuracoes versionadas por tipo de execucao
- `outputs/`: saidas consolidadas dos pipelines ativos
- `data/`: dados de entrada ativos
- `legacy/old-proj/`: codigo, configs e resultados historicos
- `legacy/backbone-analisys-historical/`: artefatos historicos de analise estrutural

## Observacao

A migracao ainda e incremental. Alguns componentes de backbone continuam sendo reaproveitados da pasta raiz `backbone/`, agora encapsulados por wrappers em `pipeline/backbone/`.

Convencao atual de saida:

- `data/npy` e `data/GraphML`: artefatos de dados reutilizaveis
- `outputs/backbone`: relatorios e manifestos gerados junto da extracao de backbone
- `outputs/analysis`: analises comparativas executadas de forma avulsa
- `outputs/forecasting`: resultados do pipeline de modelos

Estado atual da migracao de backbone:

- `pipeline/backbone/generation.py`: orquestracao principal de geracao
- `pipeline/backbone/analysis_runtime.py`: runtime principal da analise
- `pipeline/backbone/filters/`: implementacoes fisicamente consolidadas dos filtros
- `backbone/*.py`: wrappers de compatibilidade para chamadas antigas

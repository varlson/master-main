# Refat

Pipeline novo para deixar a execucao mais previsivel sem mexer no `old-proj`.

## Fluxos

- `search_best`: roda grid search apenas em datasets originais e salva as melhores configuracoes por dataset/modelo.
- `run_best`: carrega um arquivo `best_configs.json` e executa treino, previsao, relatorios e plots com os melhores parametros.
- `search_and_run`: faz busca e fase final no mesmo fluxo, ainda usando apenas datasets originais.

## Como executar

```bash
python3 -m refat.main --config refat/config.search.example.json
python3 -m refat.main --config refat/config.run_best.example.json
```

## Dry run

```bash
python3 -m refat.main --config refat/config.search.example.json --dry-run
```

## Observacoes

- A busca de hiperparametros sempre usa dados originais.
- No modo `run_best`, se o dataset alvo for backbone, o pipeline tenta reaproveitar a melhor configuracao do dataset original base, por exemplo `metr-la` para `metr-la-by-high_sal-with-alpah_filter0_4`.
- O motor de treino e os modelos continuam vindo de `old-proj`, mas a orquestracao nova fica isolada em `refat`.

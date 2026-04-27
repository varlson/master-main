# Comparacao Entre os Metodos de Backbone

## Escopo

Este documento compara os dois metodos de extração de backbone atualmente avaliados:

- `disp_fil` (`Disparity Filter`)
- `nois_corr` (`Noise Corrected Filter`)

A comparacao usa os resultados estruturais ja gerados em `backbone/analisys` para os datasets `metr-la` e `pems-bay`.

Observacao importante:

- o repositorio ainda nao possui uma pasta `results/backbone` com execucoes de predicao consolidadas;
- por isso, os pontos sobre desempenho preditivo abaixo sao hipoteses estruturais fortes, e nao confirmacoes experimentais finais;
- ainda assim, essas hipoteses sao bem fundamentadas porque os modelos espaco-temporais dependem diretamente da conectividade, da redundancia local, da preservacao dos hubs e da manutencao do componente gigante.

## Resumo Executivo

- `nois_corr` e o candidato mais promissor para predicao nos dois datasets, porque preserva todos os nos, mantem um componente gigante alto e conserva uma parcela relevante da estrutura local e global.
- `disp_fil` simplifica muito mais a rede do que `nois_corr`, a ponto de transformar os grafos em estruturas extremamente fragmentadas. Isso tende a destruir boa parte do sinal espacial que os modelos usam para prever trafego.
- A aparente reducao do comprimento medio de caminhos em `disp_fil` nao deve ser lida como ganho estrutural. Ela ocorre porque o grafo colapsa em componentes minimos, e nao porque a rede ficou mais eficiente.
- `pems-bay` parece aceitar melhor o `nois_corr` do que `metr-la`, porque o backbone resultante ainda preserva clustering alto, mais arestas e um componente gigante forte.
- `metr-la` parece mais sensivel a poda mesmo com `nois_corr`: o caminho medio mais que dobra, a eficiencia global cai bastante e a rede fica mais modular e mais alongada.

## Foto Estrutural Rapida

### `metr-la`

| Metrica | Original | `disp_fil` | `nois_corr` |
| --- | ---: | ---: | ---: |
| Nos | 207 | 138 | 207 |
| Arestas | 1313 | 17 | 378 |
| Arestas preservadas | 100.00% | 1.29% | 28.79% |
| Densidade | 0.0616 | 0.0018 | 0.0177 |
| Clustering medio | 0.5485 | 0.0000 | 0.3663 |
| Eficiencia global | 0.2789 | 0.0020 | 0.1345 |
| Caminho medio no GCC | 5.0637 | 1.3333 | 11.4514 |
| Razao do componente gigante | 0.9952 | 0.0217 | 0.9758 |
| Modularidade | 0.6978 | 0.9202 | 0.8346 |
| Assortatividade | 0.4616 | -0.2143 | 0.2368 |
| Comunidades | 8 | 121 | 20 |
| Top-k overlap medio | - | 0.1300 | 0.2400 |
| Spearman medio das centralidades | - | 0.1542 | 0.1676 |
| Robustez alvo AUC-LCC | 0.3877 | 0.0078 | 0.1543 |

### `pems-bay`

| Metrica | Original | `disp_fil` | `nois_corr` |
| --- | ---: | ---: | ---: |
| Nos | 325 | 53 | 325 |
| Arestas | 2079 | 7 | 941 |
| Arestas preservadas | 100.00% | 0.34% | 45.26% |
| Densidade | 0.0395 | 0.0051 | 0.0179 |
| Clustering medio | 0.6737 | 0.0000 | 0.5762 |
| Eficiencia global | 0.2423 | 0.0051 | 0.1385 |
| Caminho medio no GCC | 5.1616 | 1.0000 | 8.6367 |
| Razao do componente gigante | 0.9815 | 0.0377 | 0.9385 |
| Modularidade | 0.8198 | 0.8571 | 0.8941 |
| Assortatividade | 0.5696 | nan | 0.3359 |
| Comunidades | 19 | 46 | 31 |
| Top-k overlap medio | - | 0.0700 | 0.1700 |
| Spearman medio das centralidades | - | 0.1588 | 0.1981 |
| Robustez alvo AUC-LCC | 0.2687 | 0.0208 | 0.1912 |

## Por Que `nois_corr` Deve Predizer Melhor

### 1. Ele preserva os nos e o dominio do problema

Em ambos os datasets, `nois_corr` manteve todos os nos:

- `metr-la`: `207 -> 207`
- `pems-bay`: `325 -> 325`

Ja o `disp_fil` removeu muitos nos:

- `metr-la`: `207 -> 138`
- `pems-bay`: `325 -> 53`

Isso importa muito para predicao porque, no pipeline atual, quando ha remocao de nos o `h5` tambem e reduzido. Ou seja, o metodo nao esta apenas limpando a topologia: ele esta mudando o proprio conjunto de sensores/series usados pelo modelo. Isso torna o problema de predicao muito mais diferente do original.

Em termos práticos:

- `nois_corr` ainda oferece ao modelo uma rede comparavel ao problema original;
- `disp_fil` altera demais o espaco de predicao, e tende a perder informacao espacial importante.

### 2. Ele mantem conectividade suficiente para modelos espaco-temporais

Modelos como `GraphWaveNet`, `MTGNN` e variantes de attention espaco-temporal dependem de uma malha minima de conectividade para propagar sinal entre nos proximos e hubs relevantes.

O `nois_corr` manteve um componente gigante alto:

- `metr-la`: `0.9758`
- `pems-bay`: `0.9385`

O `disp_fil` praticamente colapsou o componente gigante:

- `metr-la`: `0.0217`
- `pems-bay`: `0.0377`

Isso sugere que:

- com `nois_corr`, o modelo ainda tem uma base espacial utilizavel;
- com `disp_fil`, a parte espacial do modelo tende a perder utilidade e o sistema passa a depender quase so do ramo temporal.

### 3. Ele preserva melhor redundancia local e rotas alternativas

O `nois_corr` ainda preserva clustering e eficiencia:

- `metr-la`: clustering `0.3663`, eficiencia global `0.1345`
- `pems-bay`: clustering `0.5762`, eficiencia global `0.1385`

No `disp_fil`, ambos praticamente zeram:

- `metr-la`: clustering `0.0000`, eficiencia `0.0020`
- `pems-bay`: clustering `0.0000`, eficiencia `0.0051`

Para predicao, isso importa porque a redundancia local ajuda o modelo a:

- suavizar ruído entre sensores correlacionados;
- manter caminhos alternativos de propagacao;
- generalizar melhor quando um subconjunto de conexoes nao e tao informativo.

### 4. Ele preserva melhor a hierarquia de importancia dos nos

Nenhum dos dois metodos preservou muito bem as centralidades, mas `nois_corr` foi consistentemente menos destrutivo:

- `metr-la`: overlap top-k `0.24` contra `0.13`
- `pems-bay`: overlap top-k `0.17` contra `0.07`

Isso sugere que o `nois_corr` preserva melhor quem continua sendo hub, ponte e no relevante. Para modelos com aprendizado de dependencia espacial, isso tende a produzir uma adjacencia mais coerente com a rede original.

## Por Que `disp_fil` Simplifica Muito Mais

### 1. O criterio dele e muito conservador neste contexto

Pelo codigo em [disparity_filter.py](/home/varlson/master/main/master-main/backbone/disparity_filter.py), o `Disparity Filter` calcula significancia local por no usando o peso normalizado da aresta em relacao a `strength` de cada extremidade, e depois usa o maior `alpha` entre os dois lados, o que torna o corte mais conservador.

Na pratica, isso favorece apenas arestas que dominam fortemente a distribuicao local de peso de seus nos. Em redes de trafego com muitas conexoes moderadas, isso tende a reter apenas pouquissimas arestas muito fortes.

Esse efeito aparece claramente nos pesos medios retidos:

- `metr-la`: `disp_fil = 0.9523`, `nois_corr = 0.6188`
- `pems-bay`: `disp_fil = 0.9966`, `nois_corr = 0.6923`

Ou seja, o `disp_fil` esta quase ficando apenas com as ligacoes extremas.

### 2. A modularidade sobe, mas por fragmentacao

No `disp_fil`, a modularidade cresce:

- `metr-la`: `0.6978 -> 0.9202`
- `pems-bay`: `0.8198 -> 0.8571`

So que isso nao significa necessariamente uma melhor organizacao para predicao. Aqui, o aumento vem junto com:

- `121` comunidades em `metr-la`, com maior comunidade de tamanho `3`
- `46` comunidades em `pems-bay`, com maior comunidade de tamanho `2`

Isso e um sinal claro de fragmentacao extrema, nao de separacao funcional saudavel.

### 3. O caminho medio cai porque o grafo colapsa

No `disp_fil`, o caminho medio no componente gigante ficou menor:

- `metr-la`: `5.0637 -> 1.3333`
- `pems-bay`: `5.1616 -> 1.0000`

Mas isso nao e bom sinal. O valor caiu porque o componente gigante virou uma estrutura minuscula:

- `metr-la`: componente gigante com razao `0.0217`
- `pems-bay`: componente gigante com razao `0.0377`

Entao o grafo nao ficou mais eficiente; ele apenas perdeu alcance global e sobrou uma componente trivial para medir distancia.

## Diferencas Entre os Datasets

### `metr-la`

O `metr-la` parece mais sensivel a poda global mesmo quando usamos `nois_corr`.

Sinais disso:

- caminho medio sobe `+126.14%`
- eficiencia global cai `-51.76%`
- a rede sai de `8` para `20` comunidades
- a assortatividade cai de `0.4616` para `0.2368`

Interpretacao:

- o dataset parece depender mais de atalhos e conectividade intermediaria;
- ao remover arestas, o backbone ainda preserva os nos, mas alonga demais os caminhos;
- isso pode reduzir a capacidade do modelo de capturar propagacao espacial de medio alcance.

Em outras palavras, `nois_corr` ainda e claramente melhor que `disp_fil` em `metr-la`, mas o custo estrutural da simplificacao continua alto.

### `pems-bay`

O `pems-bay` parece mais compativel com `nois_corr`.

Sinais disso:

- reducao de arestas mais moderada do que em `metr-la`: `-54.74%`
- clustering ainda alto: `0.5762`
- componente gigante segue forte: `0.9385`
- robustez alvo continua relativamente melhor: `0.1912`

Interpretacao:

- ha uma estrutura local suficientemente forte para sobreviver ao filtro;
- o backbone continua relativamente informativo para a parte espacial dos modelos;
- nesse dataset, o `nois_corr` parece mais proximo de um processo de denoising do que de colapso topologico.

Ja o `disp_fil` em `pems-bay` foi especialmente severo:

- removeu `83.69%` dos nos
- removeu `99.66%` das arestas
- deixou apenas `7` arestas

Isso sugere que o `pems-bay` possui muitas arestas que sao relevantes para a conectividade global, mas nao passam no criterio de significancia local extrema exigido pelo `disp_fil`.

## Hipoteses Mais Provaveis Para Diferenca de Predicao

Se os resultados de predicao confirmarem que `nois_corr` supera `disp_fil`, a explicacao mais consistente e:

1. `nois_corr` limpa ruído sem destruir o grafo.
2. `disp_fil` simplifica demais e remove o proprio mecanismo espacial que ajuda a prever.
3. A preservacao do componente gigante, do clustering e da robustez importa mais para predicao do que uma compressao agressiva.
4. Em `pems-bay`, o ganho relativo de `nois_corr` pode ser ainda mais claro porque a estrutura resultante permanece mais utilizavel.
5. Em `metr-la`, mesmo o `nois_corr` pode trazer queda em relacao ao grafo original, porque os caminhos ficam longos demais, embora ainda deva superar `disp_fil`.

## Conclusao

Com base nos resultados estruturais atuais, a leitura mais forte para o artigo e:

- `disp_fil` funciona como um metodo de compressao extrema, mas tende a descaracterizar demais a rede para forecasting espaco-temporal;
- `nois_corr` oferece um compromisso bem melhor entre simplificacao e preservacao do sinal estrutural;
- a simples reducao do numero de arestas nao explica sozinha a performance;
- o que mais parece importar e quanto do backbone ainda preserva conectividade global, redundancia local, hierarquia de nos centrais e robustez estrutural.

Se a proxima rodada experimental confirmar isso em `results/backbone`, a narrativa do artigo pode defender que:

- backbones uteis para predicao nao sao os mais esparsos;
- sao os que removem ruído mantendo o suficiente da geometria funcional da rede.

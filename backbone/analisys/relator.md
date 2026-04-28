# Relatorio Comparativo de Backbone

Gerado em: 2026-04-15

## Escopo

Este relatorio consolida a analise estrutural ja gerada em `backbone/analisys` para os dois datasets:

- `metr-la`
- `pems-bay`

e para os dois metodos de extracao de backbone:

- `nois_corr` (`Noise Corrected Filter`)
- `disp_fil` (`Disparity Filter`)

Todas as leituras abaixo usam os artefatos com `alpha=0.3`:

- `backbone/analisys/metr-la/metr-la-by-nois_corr-with-alpah_filter0_3`
- `backbone/analisys/metr-la/metr-la-by-disp_fil-with-alpah_filter0_3`
- `backbone/analisys/pems-bay/pems-bay-by-nois_corr-with-alpah_filter0_3`
- `backbone/analisys/pems-bay/pems-bay-by-disp_fil-with-alpah_filter0_3`

Observacao importante:

- neste relatorio, "performou melhor" significa "preservou melhor a estrutura original e a robustez da rede";
- ele nao mede MAE/RMSE de forecasting, porque esses resultados nao estao consolidados dentro de `/backbone/analisys`;
- portanto, as conclusoes abaixo sao estruturais, mas ja sao fortes o bastante para orientar a proxima rodada experimental.

## Resumo Executivo

1. `nois_corr` foi o melhor compromisso estrutural nos dois datasets, especialmente em `pems-bay`.
2. `disp_fil` podou muito mais agressivamente e deteriorou a conectividade global nas duas redes.
3. A sua suspeita esta correta, mas com uma nuance importante:
   - com `nois_corr`, `metr-la` aparenta ser mais sensivel;
   - com `disp_fil`, `pems-bay` foi muito mais sensivel.
4. Isso sugere que a sensibilidade a backbone nao depende so do dataset em si, mas da combinacao entre:
   - topologia original;
   - fragilidade das pontes entre comunidades;
   - severidade do criterio de poda.
5. Nesta rodada com `alpha=0.3`, nenhum metodo removeu nos. Isso e bom para a comparacao, porque o que muda aqui e a topologia de conectividade, e nao o conjunto de sensores.
6. O maior ganho potencial de backbone parece vir quando o metodo atua como denoising estrutural leve, nao como compressao extrema.

## Leitura Principal

### Como cada dataset performou com os metodos

#### `metr-la`

Com `nois_corr`, o `metr-la` perdeu `22.16%` das arestas, mas manteve o mesmo numero de componentes (`2`) e a mesma razao do componente gigante (`0.9952`). As perdas estruturais foram moderadas: clustering `-11.97%`, eficiencia global `-8.73%`, robustez alvo `-8.35%`. Ao mesmo tempo, a hierarquia das centralidades foi muito bem preservada (`Spearman medio = 0.9630`, `top-k medio = 0.7700`). Em termos estruturais, isso parece um backbone utilizavel.

Com `disp_fil`, o comportamento muda bastante. O metodo remove `72.28%` das arestas, aumenta os componentes de `2` para `17`, reduz a razao do componente gigante para `0.8454`, mais do que dobra o caminho medio no GCC (`+113.09%`) e derruba a eficiencia global em `-61.57%`. A robustez alvo cai `-71.74%`. Ou seja, o `metr-la` ainda retém um nucleo conectado razoavel, mas perde boa parte da sua capacidade de transporte global.

#### `pems-bay`

Com `nois_corr`, o `pems-bay` foi o melhor encaixe observado entre dataset e metodo. A poda e leve (`-14.77%` de arestas), o numero de componentes continua `7`, a razao do componente gigante fica identica (`0.9815`), o clustering cai so `-2.20%` e a eficiencia global cai `-7.87%`. A preservacao das centralidades foi excelente (`Spearman medio = 0.9848`, `menor Spearman = 0.9573`, `top-k medio = 0.8400`). O dado mais forte e que a robustez alvo melhora `+4.12%`, sugerindo que o filtro remove ruido sem destruir a espinha dorsal funcional da rede.

Com `disp_fil`, o `pems-bay` e o pior caso da rodada. Embora o metodo remova uma proporcao de arestas parecida com a de `metr-la` (`-66.57%` contra `-72.28%`), o dano estrutural e muito maior: os componentes sobem de `7` para `32`, a razao do componente gigante despenca de `0.9815` para `0.3323`, a eficiencia global cai `-84.38%` e a robustez alvo cai `-80.19%`. Em outras palavras, o `pems-bay` aceitou muito bem o `nois_corr`, mas rejeitou fortemente o `disp_fil`.

### Como os datasets combinaram ou nao com os metodos

#### Melhor combinacao: `pems-bay` + `nois_corr`

Essa foi a combinacao que mais se aproximou de um backbone "saudavel". A rede ficou mais enxuta, mas sem perder conectividade global. O backbone preservou clustering alto (`0.6589`), eficiencia local (`0.8024`), componente gigante forte (`0.9815`) e a hierarquia dos nos centrais.

#### Combinacao aceitavel, mas com custo: `metr-la` + `nois_corr`

O metodo ainda preserva bem a estrutura, mas o custo relativo em `metr-la` e maior que em `pems-bay`. O caminho medio aumenta mais, a assortatividade cai mais e a perda de clustering tambem e maior. Isso sugere que `metr-la` depende mais de arestas intermediarias para manter atalhos estruturais.

#### Combinacao ruim: `metr-la` + `disp_fil`

Aqui o backbone ainda deixa um nucleo conectado relativamente grande (`GCC = 0.8454`), mas a rede fica muito mais longa, menos eficiente e mais fragmentada. O ganho de modularidade nao compensa a perda de transporte global.

#### Pior combinacao: `pems-bay` + `disp_fil`

Foi a combinacao mais destrutiva de todas. O filtro parece reter so as conexoes locais mais dominantes, mas remove muitas pontes estruturais que sustentavam a conectividade entre comunidades. O resultado e uma rede muito mais fragmentada e com capacidade muito menor de espalhar informacao.

### O que os indices pos extracao revelam sobre os datasets originais

Os backbones ajudam a enxergar diferencas importantes entre as redes originais.

#### O `pems-bay` original parece mais clusterizado e mais modular

Na rede original, `pems-bay` tem:

- clustering medio maior: `0.6737` vs `0.5485` em `metr-la`;
- modularidade maior: `0.8198` vs `0.6978`;
- assortatividade maior: `0.5696` vs `0.4616`;
- overlap medio de arestas maior: `0.5242` vs `0.3876`.

Isso sugere uma rede mais redundante localmente, mais segmentada em comunidades e com maior homofilia estrutural. Essa estrutura combina muito bem com um filtro de denoising leve como `nois_corr`.

#### Mas o `pems-bay` original tambem parece mais dependente de pontes frageis

Apesar de ser mais clusterizado, o `pems-bay` ja nasce com `7` componentes e conectividade do GCC igual a `1` tanto por no quanto por aresta. Ja o `metr-la` tem `2` componentes e conectividade `2` no GCC. Isso e uma pista forte: o `pems-bay` pode ter muita redundancia local, mas depende de poucas pontes intercomunidade para manter a parte global unida.

Essa leitura encaixa quase perfeitamente com o comportamento do `disp_fil`: ele retira o que parece "pouco dominante" localmente, mas muitas dessas arestas podem ser exatamente as pontes globais que mantem comunidades conectadas. Por isso o `pems-bay` colapsa tanto no GCC quando podado com `disp_fil`.

#### O `metr-la` original parece menos modular, mas com um core global um pouco mais coeso

O `metr-la` tem menos clustering, menos modularidade e menos overlap medio que `pems-bay`, mas seu GCC original parece estruturalmente menos fragil a remocao de pontes, porque sua conectividade no GCC e `2`. Isso explica por que o `disp_fil` ainda machuca bastante o `metr-la`, mas nao o faz colapsar tao violentamente quanto em `pems-bay`.

### O que os pesos das arestas revelam sobre os dois metodos

Os dois metodos aumentam o peso medio e o peso mediano das arestas retidas, mas o `disp_fil` faz isso de forma muito mais radical:

- `metr-la`: peso medio `0.406 -> 0.473 -> 0.776`
- `pems-bay`: peso medio `0.530 -> 0.582 -> 0.883`

Isso mostra que o `disp_fil` esta concentrando a rede em um subconjunto pequeno de ligacoes muito fortes. Em termos interpretativos, ele funciona mais como um extrator de "arestas dominantes" do que como um denoiser equilibrado. O `nois_corr`, por outro lado, poda menos e preserva uma faixa bem mais ampla de pesos.

### O que centralidade e overlap revelam alem do obvio

Existem duas observacoes que valem destaque:

1. Em `nois_corr`, o ranking global das centralidades quase nao muda, mas os top-20 por grau mudam bem mais do que closeness, betweenness e eigenvector. Isso sugere que a poda afeta mais a hierarquia dos hubs locais do que a estrutura global de influencia.

2. Em `pems-bay` com `disp_fil`, o `Spearman` de `eigenvector_centrality` cai para `0.0476`, mas o overlap top-20 dessa mesma metrica continua `1.0`. Isso indica que os hubs principais ainda sobrevivem, porem a ordenacao relativa do resto da rede colapsa. Em outras palavras: o "miolo" dos hubs permanece, mas a malha ao redor perde coerencia.

### Modularidade maior nem sempre significa melhora

Nos dois datasets, a modularidade sobe apos a extracao, especialmente com `disp_fil`:

- `metr-la`: `0.6978 -> 0.8314`
- `pems-bay`: `0.8198 -> 0.9061`

Esse aumento nao deve ser interpretado automaticamente como ganho. Aqui ele vem junto de:

- mais componentes;
- comunidades menores;
- caminhos mais longos;
- menor eficiencia global;
- menor robustez.

Logo, a modularidade maior parece sinalizar fragmentacao, nao necessariamente uma organizacao funcional melhor.

## Tabela 1: `metr-la` - indices originais e pos-extracao

Nas colunas dos backbones, o valor entre parenteses indica o delta percentual em relacao a rede original. A tabela abaixo usa apenas os artefatos atualmente disponiveis em `backbone/analisys/metr-la`, todos com `alpha=0.3`.

| Metrica | Original | Noise Corrected (alpha=0.3) | Disparity Filter (alpha=0.3) | High Salience (alpha=0.3) |
| --- | ---: | ---: | ---: | ---: |
| Nos | 207 | 207 (Delta +0.00%) | 207 (Delta +0.00%) | 128 (Delta -38.16%) |
| Arestas | 1313 | 1022 (Delta -22.16%) | 430 (Delta -67.25%) | 97 (Delta -92.61%) |
| Densidade | 0.0616 | 0.0479 (Delta -22.16%) | 0.0202 (Delta -67.25%) | 0.0119 (Delta -80.62%) |
| Grau medio | 12.69 | 9.87 (Delta -22.16%) | 4.15 (Delta -67.25%) | 1.52 (Delta -88.05%) |
| Forca media | 5.14 | 4.67 (Delta -9.17%) | 3.07 (Delta -40.33%) | 0.91 (Delta -82.32%) |
| Peso medio | 0.406 | 0.473 (Delta +16.70%) | 0.739 (Delta +82.19%) | 0.600 (Delta +47.97%) |
| Peso mediano | 0.321 | 0.421 (Delta +30.96%) | 0.727 (Delta +126.41%) | 0.612 (Delta +90.50%) |
| Clustering medio | 0.5485 | 0.4829 (Delta -11.97%) | 0.4157 (Delta -24.22%) | 0.0000 (Delta -100.00%) |
| Eficiencia local | 0.7473 | 0.6985 (Delta -6.53%) | 0.5044 (Delta -32.49%) | 0.0000 (Delta -100.00%) |
| Eficiencia global | 0.2789 | 0.2545 (Delta -8.73%) | 0.1319 (Delta -52.71%) | 0.0246 (Delta -91.16%) |
| Componentes | 2 | 2 (Delta +0.00%) | 12 (Delta +500.00%) | 31 (Delta +1450.00%) |
| Razao GCC | 0.9952 | 0.9952 (Delta +0.00%) | 0.9034 (Delta -9.22%) | 0.1797 (Delta -81.94%) |
| Caminho medio GCC | 5.0637 | 5.5292 (Delta +9.19%) | 9.9274 (Delta +96.05%) | 4.3320 (Delta -14.45%) |
| Diametro GCC | 13.0 | 14.0 (Delta +7.69%) | 27.0 (Delta +107.69%) | 10.0 (Delta -23.08%) |
| Assortatividade | 0.4616 | 0.2622 (Delta -43.20%) | 0.5375 (Delta +16.43%) | -0.0949 (Delta -120.56%) |
| Comunidades | 8 | 8 (Delta +0.00%) | 24 (Delta +200.00%) | 34 (Delta +325.00%) |
| Modularidade | 0.6978 | 0.7094 (Delta +1.66%) | 0.8055 (Delta +15.43%) | 0.9373 (Delta +34.33%) |
| Maior comunidade | 53 | 53 (Delta +0.00%) | 26 (Delta -50.94%) | 9 (Delta -83.02%) |

Script LaTeX da tabela acima:

```latex
\begin{table}[ht]
\centering
\caption{metr-la: indices estruturais originais e apos extracao de backbone para os artefatos atualmente disponiveis em backbone/analisys/metr-la (todos com alpha=0.3), com delta percentual em relacao a rede original.}
\label{tab:metr_la_indices}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrr}
\toprule
Metrica & Original & Noise Corrected (alpha=0.3) & Disparity Filter (alpha=0.3) & High Salience (alpha=0.3) \\
\midrule
Nos & 207 & 207 ($\Delta$ +0.00\%) & 207 ($\Delta$ +0.00\%) & 128 ($\Delta$ -38.16\%) \\
Arestas & 1313 & 1022 ($\Delta$ -22.16\%) & 430 ($\Delta$ -67.25\%) & 97 ($\Delta$ -92.61\%) \\
Densidade & 0.0616 & 0.0479 ($\Delta$ -22.16\%) & 0.0202 ($\Delta$ -67.25\%) & 0.0119 ($\Delta$ -80.62\%) \\
Grau medio & 12.69 & 9.87 ($\Delta$ -22.16\%) & 4.15 ($\Delta$ -67.25\%) & 1.52 ($\Delta$ -88.05\%) \\
Forca media & 5.14 & 4.67 ($\Delta$ -9.17\%) & 3.07 ($\Delta$ -40.33\%) & 0.91 ($\Delta$ -82.32\%) \\
Peso medio & 0.406 & 0.473 ($\Delta$ +16.70\%) & 0.739 ($\Delta$ +82.19\%) & 0.600 ($\Delta$ +47.97\%) \\
Peso mediano & 0.321 & 0.421 ($\Delta$ +30.96\%) & 0.727 ($\Delta$ +126.41\%) & 0.612 ($\Delta$ +90.50\%) \\
Clustering medio & 0.5485 & 0.4829 ($\Delta$ -11.97\%) & 0.4157 ($\Delta$ -24.22\%) & 0.0000 ($\Delta$ -100.00\%) \\
Eficiencia local & 0.7473 & 0.6985 ($\Delta$ -6.53\%) & 0.5044 ($\Delta$ -32.49\%) & 0.0000 ($\Delta$ -100.00\%) \\
Eficiencia global & 0.2789 & 0.2545 ($\Delta$ -8.73\%) & 0.1319 ($\Delta$ -52.71\%) & 0.0246 ($\Delta$ -91.16\%) \\
Componentes & 2 & 2 ($\Delta$ +0.00\%) & 12 ($\Delta$ +500.00\%) & 31 ($\Delta$ +1450.00\%) \\
Razao GCC & 0.9952 & 0.9952 ($\Delta$ +0.00\%) & 0.9034 ($\Delta$ -9.22\%) & 0.1797 ($\Delta$ -81.94\%) \\
Caminho medio GCC & 5.0637 & 5.5292 ($\Delta$ +9.19\%) & 9.9274 ($\Delta$ +96.05\%) & 4.3320 ($\Delta$ -14.45\%) \\
Diametro GCC & 13.0 & 14.0 ($\Delta$ +7.69\%) & 27.0 ($\Delta$ +107.69\%) & 10.0 ($\Delta$ -23.08\%) \\
Assortatividade & 0.4616 & 0.2622 ($\Delta$ -43.20\%) & 0.5375 ($\Delta$ +16.43\%) & -0.0949 ($\Delta$ -120.56\%) \\
Comunidades & 8 & 8 ($\Delta$ +0.00\%) & 24 ($\Delta$ +200.00\%) & 34 ($\Delta$ +325.00\%) \\
Modularidade & 0.6978 & 0.7094 ($\Delta$ +1.66\%) & 0.8055 ($\Delta$ +15.43\%) & 0.9373 ($\Delta$ +34.33\%) \\
Maior comunidade & 53 & 53 ($\Delta$ +0.00\%) & 26 ($\Delta$ -50.94\%) & 9 ($\Delta$ -83.02\%) \\
\bottomrule
\end{tabular}%
}
\end{table}
```

## Tabela 2: `pems-bay` - indices originais e pos-extracao

Nas colunas dos backbones, o valor entre parenteses indica o delta percentual em relacao a rede original. Como `pems-bay` possui dois artefatos de `disp_fil` atualmente disponiveis em `backbone/analisys/pems-bay`, ambos sao mostrados abaixo.

| Metrica | Original | Noise Corrected (alpha=0.3) | Disparity Filter (alpha=0.3) | Disparity Filter (alpha=0.8) | High Salience (alpha=0.3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nos | 325 | 325 (Delta +0.00%) | 325 (Delta +0.00%) | 325 (Delta +0.00%) | 192 (Delta -40.92%) |
| Arestas | 2079 | 1772 (Delta -14.77%) | 858 (Delta -58.73%) | 2037 (Delta -2.02%) | 171 (Delta -91.77%) |
| Densidade | 0.0395 | 0.0337 (Delta -14.77%) | 0.0163 (Delta -58.73%) | 0.0387 (Delta -2.02%) | 0.0093 (Delta -76.38%) |
| Grau medio | 12.79 | 10.90 (Delta -14.77%) | 5.28 (Delta -58.73%) | 12.54 (Delta -2.02%) | 1.78 (Delta -86.08%) |
| Forca media | 6.78 | 6.35 (Delta -6.38%) | 4.43 (Delta -34.63%) | 6.75 (Delta -0.42%) | 1.15 (Delta -83.03%) |
| Peso medio | 0.530 | 0.582 (Delta +9.84%) | 0.840 (Delta +58.40%) | 0.539 (Delta +1.64%) | 0.646 (Delta +21.92%) |
| Peso mediano | 0.509 | 0.588 (Delta +15.67%) | 0.875 (Delta +72.06%) | 0.516 (Delta +1.42%) | 0.658 (Delta +29.32%) |
| Clustering medio | 0.6737 | 0.6589 (Delta -2.20%) | 0.6493 (Delta -3.62%) | 0.6756 (Delta +0.28%) | 0.0000 (Delta -100.00%) |
| Eficiencia local | 0.8130 | 0.8024 (Delta -1.30%) | 0.7509 (Delta -7.64%) | 0.8144 (Delta +0.18%) | 0.0000 (Delta -100.00%) |
| Eficiencia global | 0.2423 | 0.2233 (Delta -7.87%) | 0.0536 (Delta -77.87%) | 0.2392 (Delta -1.32%) | 0.0417 (Delta -82.78%) |
| Componentes | 7 | 7 (Delta +0.00%) | 21 (Delta +200.00%) | 7 (Delta +0.00%) | 21 (Delta +200.00%) |
| Razao GCC | 0.9815 | 0.9815 (Delta +0.00%) | 0.4215 (Delta -57.05%) | 0.9815 (Delta +0.00%) | 0.4688 (Delta -52.24%) |
| Caminho medio GCC | 5.1616 | 5.6190 (Delta +8.86%) | 7.7657 (Delta +50.45%) | 5.2587 (Delta +1.88%) | 12.3094 (Delta +138.48%) |
| Diametro GCC | 13.0 | 13.0 (Delta +0.00%) | 23.0 (Delta +76.92%) | 13.0 (Delta +0.00%) | 33.0 (Delta +153.85%) |
| Assortatividade | 0.5696 | 0.4748 (Delta -16.63%) | 0.7447 (Delta +30.74%) | 0.5762 (Delta +1.16%) | -0.1255 (Delta -122.03%) |
| Comunidades | 19 | 20 (Delta +5.26%) | 32 (Delta +68.42%) | 19 (Delta +0.00%) | 28 (Delta +47.37%) |
| Modularidade | 0.8198 | 0.8425 (Delta +2.76%) | 0.8977 (Delta +9.49%) | 0.8205 (Delta +0.08%) | 0.9144 (Delta +11.53%) |
| Maior comunidade | 67 | 53 (Delta -20.90%) | 42 (Delta -37.31%) | 67 (Delta +0.00%) | 17 (Delta -74.63%) |

Script LaTeX da tabela acima:

```latex
\begin{table}[ht]
\centering
\caption{pems-bay: indices estruturais originais e apos extracao de backbone para os artefatos atualmente disponiveis em backbone/analisys/pems-bay, com delta percentual em relacao a rede original.}
\label{tab:pems_bay_indices}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
Metrica & Original & Noise Corrected (alpha=0.3) & Disparity Filter (alpha=0.3) & Disparity Filter (alpha=0.8) & High Salience (alpha=0.3) \\
\midrule
Nos & 325 & 325 ($\Delta$ +0.00\%) & 325 ($\Delta$ +0.00\%) & 325 ($\Delta$ +0.00\%) & 192 ($\Delta$ -40.92\%) \\
Arestas & 2079 & 1772 ($\Delta$ -14.77\%) & 858 ($\Delta$ -58.73\%) & 2037 ($\Delta$ -2.02\%) & 171 ($\Delta$ -91.77\%) \\
Densidade & 0.0395 & 0.0337 ($\Delta$ -14.77\%) & 0.0163 ($\Delta$ -58.73\%) & 0.0387 ($\Delta$ -2.02\%) & 0.0093 ($\Delta$ -76.38\%) \\
Grau medio & 12.79 & 10.90 ($\Delta$ -14.77\%) & 5.28 ($\Delta$ -58.73\%) & 12.54 ($\Delta$ -2.02\%) & 1.78 ($\Delta$ -86.08\%) \\
Forca media & 6.78 & 6.35 ($\Delta$ -6.38\%) & 4.43 ($\Delta$ -34.63\%) & 6.75 ($\Delta$ -0.42\%) & 1.15 ($\Delta$ -83.03\%) \\
Peso medio & 0.530 & 0.582 ($\Delta$ +9.84\%) & 0.840 ($\Delta$ +58.40\%) & 0.539 ($\Delta$ +1.64\%) & 0.646 ($\Delta$ +21.92\%) \\
Peso mediano & 0.509 & 0.588 ($\Delta$ +15.67\%) & 0.875 ($\Delta$ +72.06\%) & 0.516 ($\Delta$ +1.42\%) & 0.658 ($\Delta$ +29.32\%) \\
Clustering medio & 0.6737 & 0.6589 ($\Delta$ -2.20\%) & 0.6493 ($\Delta$ -3.62\%) & 0.6756 ($\Delta$ +0.28\%) & 0.0000 ($\Delta$ -100.00\%) \\
Eficiencia local & 0.8130 & 0.8024 ($\Delta$ -1.30\%) & 0.7509 ($\Delta$ -7.64\%) & 0.8144 ($\Delta$ +0.18\%) & 0.0000 ($\Delta$ -100.00\%) \\
Eficiencia global & 0.2423 & 0.2233 ($\Delta$ -7.87\%) & 0.0536 ($\Delta$ -77.87\%) & 0.2392 ($\Delta$ -1.32\%) & 0.0417 ($\Delta$ -82.78\%) \\
Componentes & 7 & 7 ($\Delta$ +0.00\%) & 21 ($\Delta$ +200.00\%) & 7 ($\Delta$ +0.00\%) & 21 ($\Delta$ +200.00\%) \\
Razao GCC & 0.9815 & 0.9815 ($\Delta$ +0.00\%) & 0.4215 ($\Delta$ -57.05\%) & 0.9815 ($\Delta$ +0.00\%) & 0.4688 ($\Delta$ -52.24\%) \\
Caminho medio GCC & 5.1616 & 5.6190 ($\Delta$ +8.86\%) & 7.7657 ($\Delta$ +50.45\%) & 5.2587 ($\Delta$ +1.88\%) & 12.3094 ($\Delta$ +138.48\%) \\
Diametro GCC & 13.0 & 13.0 ($\Delta$ +0.00\%) & 23.0 ($\Delta$ +76.92\%) & 13.0 ($\Delta$ +0.00\%) & 33.0 ($\Delta$ +153.85\%) \\
Assortatividade & 0.5696 & 0.4748 ($\Delta$ -16.63\%) & 0.7447 ($\Delta$ +30.74\%) & 0.5762 ($\Delta$ +1.16\%) & -0.1255 ($\Delta$ -122.03\%) \\
Comunidades & 19 & 20 ($\Delta$ +5.26\%) & 32 ($\Delta$ +68.42\%) & 19 ($\Delta$ +0.00\%) & 28 ($\Delta$ +47.37\%) \\
Modularidade & 0.8198 & 0.8425 ($\Delta$ +2.76\%) & 0.8977 ($\Delta$ +9.49\%) & 0.8205 ($\Delta$ +0.08\%) & 0.9144 ($\Delta$ +11.53\%) \\
Maior comunidade & 67 & 53 ($\Delta$ -20.90\%) & 42 ($\Delta$ -37.31\%) & 67 ($\Delta$ +0.00\%) & 17 ($\Delta$ -74.63\%) \\
\bottomrule
\end{tabular}%
}
\end{table}
```

## Tabela 3: deltas comparativos por dataset e metodo

| Dataset | Metodo | Arestas removidas (%) | Delta clustering (%) | Delta efic. global (%) | Delta caminho GCC (%) | Delta GCC (%) | Spearman medio | Top-k medio | Delta robustez alvo (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metr-la | Noise Corrected | 22.16 | -11.97 | -8.73 | +9.19 | +0.00 | 0.9630 | 0.7700 | -8.35 |
| metr-la | Disparity Filter | 72.28 | -28.65 | -61.57 | +113.09 | -15.05 | 0.7633 | 0.6600 | -71.74 |
| pems-bay | Noise Corrected | 14.77 | -2.20 | -7.87 | +8.86 | +0.00 | 0.9848 | 0.8400 | +4.12 |
| pems-bay | Disparity Filter | 66.57 | -10.42 | -84.38 | +56.57 | -66.14 | 0.4809 | 0.5100 | -80.19 |

Script LaTeX da tabela acima:

```latex
\begin{table}[ht]
\centering
\caption{Comparacao resumida dos deltas estruturais apos extracao de backbone (alpha=0.3).}
\label{tab:backbone_deltas}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrr}
\toprule
Dataset & Metodo & Arestas removidas (\%) & $\Delta$ Clustering (\%) & $\Delta$ Efic. global (\%) & $\Delta$ Caminho GCC (\%) & $\Delta$ GCC (\%) & Spearman medio & Top-k medio & $\Delta$ Robustez alvo (\%) \\
\midrule
metr-la & Noise Corrected & 22.16 & -11.97 & -8.73 & +9.19 & +0.00 & 0.9630 & 0.7700 & -8.35 \\
metr-la & Disparity Filter & 72.28 & -28.65 & -61.57 & +113.09 & -15.05 & 0.7633 & 0.6600 & -71.74 \\
pems-bay & Noise Corrected & 14.77 & -2.20 & -7.87 & +8.86 & +0.00 & 0.9848 & 0.8400 & +4.12 \\
pems-bay & Disparity Filter & 66.57 & -10.42 & -84.38 & +56.57 & -66.14 & 0.4809 & 0.5100 & -80.19 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

## Observacoes e Interpretacoes Adicionais

### 1. Backbone util nao e o backbone mais esparso

O resultado mais importante talvez seja este: reduzir muito arestas nao equivale a melhorar a rede para uso posterior. O `disp_fil` produz grafos bem mais compactos, mas o preco e alto demais em conectividade, eficiencia e robustez.

### 2. O ganho de backbone depende fortemente da topologia original

Os resultados sustentam a hipotese de que "nem sempre backbone ajuda". O metodo ajuda quando remove redundancia sem destruir as pontes estruturais que sustentam a propagacao global. Isso aconteceu melhor em `pems-bay` com `nois_corr`.

### 3. Sensibilidade a poda nao foi uniforme entre os datasets

Essa e a nuance central da rodada:

- `metr-la` foi mais sensivel ao `nois_corr`;
- `pems-bay` foi muito mais sensivel ao `disp_fil`.

Portanto, nao parece correto dizer apenas "dataset X e mais sensivel a poda". O mais correto e dizer:

- a sensibilidade depende do tipo de poda;
- a resposta do dataset depende da estrutura original da rede.

### 4. O `nois_corr` se comporta mais como denoising estrutural

Ele reduz arestas, mas preserva bem:

- o componente gigante;
- a ordem relativa das centralidades;
- a redundancia local;
- a robustez global.

Isso vale especialmente para `pems-bay`, onde o filtro chega a melhorar a robustez alvo.

### 5. O `disp_fil` se comporta mais como extracao de ligacoes dominantes

Ele concentra a rede nas arestas mais fortes e aumenta bastante peso medio e mediano das ligacoes retidas. O problema e que isso parece eliminar muitas conexoes intermediarias que, embora nao dominem localmente, sao importantes globalmente.

### 6. Para forecasting espaco-temporal, o risco principal nao e so perder arestas

O maior risco e perder:

- caminhos alternativos;
- pontes intercomunidade;
- estabilidade da hierarquia espacial;
- robustez da propagacao.

Esses sao exatamente os pontos em que o `disp_fil` falha mais forte nesta rodada.

## Conclusao

Com base nos indices estruturais desta rodada (`alpha=0.3`), a leitura mais consistente e:

1. `nois_corr` foi superior a `disp_fil` nos dois datasets.
2. `pems-bay` combinou muito bem com `nois_corr`, mostrando sinais de denoising real sem perda forte da geometria funcional.
3. `metr-la` tambem aceita `nois_corr`, mas com custo estrutural mais visivel.
4. `disp_fil` nao parece um bom candidato para uso direto como backbone de suporte a modelos espaco-temporais nesses dados, especialmente em `pems-bay`.
5. Os beneficios de backbone realmente dependem da estrutura da rede original; em especial, dependem de quanto a conectividade global repousa sobre arestas que podem parecer localmente pouco relevantes.

Se a proxima etapa for cruzar isso com forecasting, a prioridade natural e:

- comparar `original` vs `nois_corr` primeiro;
- manter muito cuidado com `disp_fil`;
- avaliar limiares de poda por dataset, em vez de assumir um mesmo nivel de agressividade para todos.

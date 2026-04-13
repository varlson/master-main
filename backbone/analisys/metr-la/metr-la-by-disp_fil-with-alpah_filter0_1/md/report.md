# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-09 00:19:25

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/metr-la/metr-la-by-disp_fil-with-alpah_filter0_1`

## Dataset: metr-la

### Rede Original

- Nos: 207 | Arestas: 1313 | Densidade: 0.0616
- Grau medio: 12.6860 | Forca media: 5.1448
- Clustering medio: 0.5485 | Eficiencia global: 0.2789 | Modularidade: 0.6978
- Caminho medio (gcc): 5.0637 | Componentes: 2 | Assortatividade: 0.4616

- Comunidades: 8 | Maior comunidade: 53 | Razao do componente gigante: 0.9952

### Comparacoes com a Rede Original

#### metr-la-by-disp_fil-with-alpah_filter0_1

Informacoes basicas do backbone:
- Nos: 138 | Delta: -33.33%
- Arestas: 17 | Delta: -98.71%
- Densidade: 0.0018 | Delta: -97.08%
- Grau medio: 0.2464 | Forca media: 0.2346
- Componentes: 121 | Razao do componente gigante: 0.0217 | Delta GCC: -97.82%
- Clustering medio: 0.0000 | Delta: -100.00%
- Eficiencia global: 0.0020 | Delta: -99.30%
- Caminho medio (gcc): 1.3333 | Delta: -73.67%
- Modularidade: 0.9202 | Delta: +0.2224
- Assortatividade: -0.2143 | Delta: -0.6759
- Comunidades: 121 | Maior comunidade: 3
- Robustez aleatoria (AUC LCC): 0.0160 | Delta: -96.49%
- Robustez alvo (AUC LCC): 0.0078 | Delta: -97.99%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.1542
- Menor correlacao de centralidade entre as metricas: -0.0568
- Overlap medio top-k: 0.1300
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - encurtou os caminhos médios no componente gigante
  - acentuou a separação entre comunidades
  - alterou de forma relevante quais nós aparecem como mais centrais
  - ficou mais frágil a ataques direcionados em hubs


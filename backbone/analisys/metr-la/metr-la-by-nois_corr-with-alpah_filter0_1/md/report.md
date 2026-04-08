# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-08 13:43:24

Pasta de saida: `/home/varlson/master/main/master-main/backbone/analisys/metr-la/metr-la-by-nois_corr-with-alpah_filter0_1`

## Dataset: metr-la

### Rede Original

- Nos: 207 | Arestas: 1313 | Densidade: 0.0616
- Clustering medio: 0.5485 | Eficiencia global: 0.2789 | Modularidade: 0.6978
- Caminho medio (gcc): 5.0637 | Componentes: 2 | Assortatividade: 0.4616

### Comparacoes com a Rede Original

#### metr-la-by-nois_corr-with-alpah_filter0_1

- Reducao de arestas: 71.21%
- Delta de densidade: -71.21%
- Delta de clustering medio: -33.23%
- Delta de caminho medio: 126.14%
- Delta de modularidade: 0.1368
- Correlacao media das centralidades: 0.1676
- Delta de robustez alvo (AUC do componente gigante): -60.19%
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - alterou de forma relevante quais nós aparecem como mais centrais
  - ficou mais frágil a ataques direcionados em hubs


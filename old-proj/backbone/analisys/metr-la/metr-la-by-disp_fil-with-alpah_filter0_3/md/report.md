# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-24 21:24:48

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/metr-la/metr-la-by-disp_fil-with-alpah_filter0_3`

## Dataset: metr-la

### Rede Original

- Nos: 207 | Arestas: 1313 | Densidade: 0.0616
- Grau medio: 12.6860 | Forca media: 5.1448
- Clustering medio: 0.5485 | Eficiencia global: 0.2789 | Modularidade: 0.6978
- Caminho medio (gcc): 5.0637 | Componentes: 2 | Assortatividade: 0.4616

- Comunidades: 8 | Maior comunidade: 53 | Razao do componente gigante: 0.9952

### Comparacoes com a Rede Original

#### metr-la-by-disp_fil-with-alpah_filter0_3

Informacoes basicas do backbone:
- Nos: 207 | Delta: -0.00%
- Arestas: 430 | Delta: -67.25%
- Densidade: 0.0202 | Delta: -67.25%
- Grau medio: 4.1546 | Forca media: 3.0697
- Componentes: 12 | Razao do componente gigante: 0.9034 | Delta GCC: -9.22%
- Clustering medio: 0.4157 | Delta: -24.22%
- Eficiencia global: 0.1319 | Delta: -52.71%
- Caminho medio (gcc): 9.9274 | Delta: +96.05%
- Modularidade: 0.8055 | Delta: +0.1077
- Assortatividade: 0.5375 | Delta: +0.0759
- Comunidades: 24 | Maior comunidade: 26
- Robustez aleatoria (AUC LCC): 0.2662 | Delta: -41.61%
- Robustez alvo (AUC LCC): 0.1373 | Delta: -64.58%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.8194
- Menor correlacao de centralidade entre as metricas: 0.6714
- Overlap medio top-k: 0.6600
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - ficou mais frágil a ataques direcionados em hubs


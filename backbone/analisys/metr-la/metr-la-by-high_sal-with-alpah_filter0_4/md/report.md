# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-22 14:24:15

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/metr-la/metr-la-by-high_sal-with-alpah_filter0_4`

## Dataset: metr-la

### Rede Original

- Nos: 207 | Arestas: 1313 | Densidade: 0.0616
- Grau medio: 12.6860 | Forca media: 5.1448
- Clustering medio: 0.5485 | Eficiencia global: 0.2789 | Modularidade: 0.6978
- Caminho medio (gcc): 5.0637 | Componentes: 2 | Assortatividade: 0.4616

- Comunidades: 8 | Maior comunidade: 53 | Razao do componente gigante: 0.9952

### Comparacoes com a Rede Original

#### metr-la-by-high_sal-with-alpah_filter0_4

Informacoes basicas do backbone:
- Nos: 168 | Delta: -18.84%
- Arestas: 139 | Delta: -89.41%
- Densidade: 0.0099 | Delta: -83.91%
- Grau medio: 1.6548 | Forca media: 1.0026
- Componentes: 29 | Razao do componente gigante: 0.4524 | Delta GCC: -54.54%
- Clustering medio: 0.0000 | Delta: -100.00%
- Eficiencia global: 0.0388 | Delta: -86.07%
- Caminho medio (gcc): 11.0053 | Delta: +117.33%
- Modularidade: 0.9084 | Delta: +0.2106
- Assortatividade: -0.0271 | Delta: -0.4887
- Comunidades: 36 | Maior comunidade: 17
- Robustez aleatoria (AUC LCC): 0.0824 | Delta: -81.92%
- Robustez alvo (AUC LCC): 0.0304 | Delta: -92.17%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.1415
- Menor correlacao de centralidade entre as metricas: -0.2747
- Overlap medio top-k: 0.3000
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - alterou de forma relevante quais nós aparecem como mais centrais
  - ficou mais frágil a ataques direcionados em hubs


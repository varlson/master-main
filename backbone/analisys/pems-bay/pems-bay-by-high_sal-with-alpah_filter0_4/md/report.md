# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-22 14:24:45

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/pems-bay/pems-bay-by-high_sal-with-alpah_filter0_4`

## Dataset: pems-bay

### Rede Original

- Nos: 325 | Arestas: 2079 | Densidade: 0.0395
- Grau medio: 12.7938 | Forca media: 6.7824
- Clustering medio: 0.6737 | Eficiencia global: 0.2423 | Modularidade: 0.8198
- Caminho medio (gcc): 5.1616 | Componentes: 7 | Assortatividade: 0.5696

- Comunidades: 19 | Maior comunidade: 67 | Razao do componente gigante: 0.9815

### Comparacoes com a Rede Original

#### pems-bay-by-high_sal-with-alpah_filter0_4

Informacoes basicas do backbone:
- Nos: 235 | Delta: -27.69%
- Arestas: 221 | Delta: -89.37%
- Densidade: 0.0080 | Delta: -79.64%
- Grau medio: 1.8809 | Forca media: 1.2202
- Componentes: 15 | Razao do componente gigante: 0.6426 | Delta GCC: -34.54%
- Clustering medio: 0.0000 | Delta: -100.00%
- Eficiencia global: 0.0602 | Delta: -75.15%
- Caminho medio (gcc): 12.0888 | Delta: +134.21%
- Modularidade: 0.9090 | Delta: +0.0892
- Assortatividade: -0.2428 | Delta: -0.8124
- Comunidades: 27 | Maior comunidade: 18
- Robustez aleatoria (AUC LCC): 0.1208 | Delta: -72.04%
- Robustez alvo (AUC LCC): 0.0305 | Delta: -88.64%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.1524
- Menor correlacao de centralidade entre as metricas: -0.2839
- Overlap medio top-k: 0.1500
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - alterou de forma relevante quais nós aparecem como mais centrais
  - ficou mais frágil a ataques direcionados em hubs


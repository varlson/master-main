# Relatorio de Analise Estrutural de Redes e Backbones

Gerado em: 2026-04-24 21:25:41

Pasta de saida: `/home/suleimane/master/main/master-main/backbone/analisys/pems-bay/pems-bay-by-high_sal-with-alpah_filter0_3`

## Dataset: pems-bay

### Rede Original

- Nos: 325 | Arestas: 2079 | Densidade: 0.0395
- Grau medio: 12.7938 | Forca media: 6.7824
- Clustering medio: 0.6737 | Eficiencia global: 0.2423 | Modularidade: 0.8198
- Caminho medio (gcc): 5.1616 | Componentes: 7 | Assortatividade: 0.5696

- Comunidades: 19 | Maior comunidade: 67 | Razao do componente gigante: 0.9815

### Comparacoes com a Rede Original

#### pems-bay-by-high_sal-with-alpah_filter0_3

Informacoes basicas do backbone:
- Nos: 192 | Delta: -40.92%
- Arestas: 171 | Delta: -91.77%
- Densidade: 0.0093 | Delta: -76.38%
- Grau medio: 1.7812 | Forca media: 1.1513
- Componentes: 21 | Razao do componente gigante: 0.4688 | Delta GCC: -52.24%
- Clustering medio: 0.0000 | Delta: -100.00%
- Eficiencia global: 0.0417 | Delta: -82.78%
- Caminho medio (gcc): 12.3094 | Delta: +138.48%
- Modularidade: 0.9144 | Delta: +0.0946
- Assortatividade: -0.1255 | Delta: -0.6951
- Comunidades: 28 | Maior comunidade: 17
- Robustez aleatoria (AUC LCC): 0.0770 | Delta: -82.18%
- Robustez alvo (AUC LCC): 0.0294 | Delta: -89.05%

Comparacao direta com a rede original:
- Correlacao media das centralidades: 0.1384
- Menor correlacao de centralidade entre as metricas: -0.4135
- Overlap medio top-k: 0.1300
- Leitura interpretativa:
  - removeu uma fração grande das arestas e tornou a rede bem mais esparsa
  - reduziu fortemente a redundância local e os triângulos
  - aumentou os caminhos médios, sugerindo perda de atalhos estruturais
  - acentuou a separação entre comunidades
  - alterou de forma relevante quais nós aparecem como mais centrais
  - ficou mais frágil a ataques direcionados em hubs


# Sobre o Projeto: Forecasting Espaço-Temporal em Redes de Tráfego

Este projeto é uma plataforma de experimentação e benchmarking para modelos de **Previsão de Séries Temporais em Grafos (Spatio-Temporal Graph Neural Networks)**, focada no domínio de tráfego urbano.

## 🎯 Objetivo

Este trabalho tem como objetivo avaliar diferentes métodos de extração de backbone em redes de séries temporais, visando identificar as técnicas que minimizam o erro de predição em comparação aos dados originais. Para a validação experimental, serão utilizados modelos de redes neurais gráficas (GNNs) aplicados aos conjuntos de dados METR-LA e PEMS-BAY.

## 🏗️ Estrutura do Projeto

O repositório está organizado da seguinte forma:

- **`main.py` / `simple_pipeline.py`**: Pontos de entrada para execução dos experimentos.
- **`models/`**: Implementações dos modelos suportados:
  - `DCRNN` (Diffusion Convolutional Recurrent Neural Network)
  - `GraphWaveNet` (WaveNet adaptada para grafos)
  - `MTGNN` (Multivariate Time Series Forecasting with GNN)
  - `DGCRN` (Dynamic Graph Convolutional Recurrent Network)
  - `STICformer` (Spatio-Temporal Interactive Cognitive Transformer)
  - `PatchSTG` (Patch-based Spatio-Temporal Graph Learning)
- **`shared/`**: Módulos fundamentais que garantem a consistência dos experimentos:
  - `loaders.py`: Processamento de dados e criação de DataLoaders.
  - `metrics.py`: Cálculo de métricas (MAE, RMSE, sMAPE, WAPE) em escala original.
  - `MLFlow.py`: Integração com MLflow para rastreamento de experimentos e grid search.
  - `reproducibility.py`: Controle de sementes (seeds) para resultados determinísticos.
  - `resultSumarization.py`: Consolidação de resultados em relatórios científicos.
- **`backbone/`**: Ferramentas para filtragem e extração de estruturas principais (backbones) de grafos, como _Disparity Filter_ e _High Salience Skeleton_.
- **`data/`**: Armazenamento dos datasets em formato `.npy`.
- **`results/`**: Saída de todos os experimentos, incluindo logs, checkpoints de modelos, arquivos CSV/JSON e visualizações (plots).

## 📊 Datasets Suportados

1. **METR-LA**: Estatísticas de tráfego de Los Angeles.
2. **PEMS-BAY**: Dados de sensores de tráfego da Bay Area (Califórnia).

## 🚀 Fluxo de Execução (Pipeline)

Para cada dataset e modelo configurado, o sistema executa:

1. **Preparação**: Carregamento, normalização e janela temporal (janela de entrada vs. horizonte de previsão).
2. **Grid Search**: Busca exaustiva de hiperparâmetros rastreada via **MLflow**.
3. **Seleção**: Escolha da melhor configuração baseada na performance no conjunto de **validação**.
4. **Teste Final**: Execução da melhor configuração com múltiplas sementes para garantir robustez estatística.
5. **Consolidação**: Geração de relatórios em Markdown, tabelas CSV e gráficos de diagnóstico (Real vs. Predito, Erro por Horizonte, etc.).

## 🛠️ Tecnologias Utilizadas

- **Linguagem**: Python 3.10+
- **Deep Learning**: PyTorch
- **Rastreamento de Experimentos**: MLflow
- **Processamento de Dados**: NumPy, Pandas, Scikit-learn
- **Visualização**: Matplotlib, Seaborn

## 📈 Diferenciais

- **Rigor Científico**: A seleção de modelos é feita rigorosamente via validação, e o teste final utiliza métricas em escala original com agregação estatística (média, desvio padrão e IC95%).
- **Extensibilidade**: Facilidade para adicionar novos modelos e datasets seguindo as interfaces padronizadas na pasta `shared/`.
- **Automatização**: Relatórios consolidados "all-in-one" que facilitam a análise comparativa direta entre múltiplos modelos e datasets em uma única rodada.

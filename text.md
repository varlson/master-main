Considerando objetivo do meu trablho, que consiste em comparar diferentes metodos de extração de backbone a fim de definir a que melhor obteve menos erro de predição de series temporais, tambem tento mostrar que, uso de backbone pode ser benefico assim como nao, dependendo do dataset, metodo de extração de backbone assim como o modelo. Considerando uso de Modelos STICformer, MTGNN e GraphWaveNet. Dois datasets metr-la e pemsbay e 3 metodos de extração de backbones Noise corrected, disparity filter e High Salience Skeleton. Estou usando três metricas, MAE RMSE e WAPE (e tambem Desta usando MAE como metrica principal). Desejo usar seguintes recursos estatiscos para testes estatisticos e a visualização.

- Radar chart, Teste de Friedman, Teste de Nemenyi e Critical Difference Diagram.

Com base nessa informação, me sugira as comparações tanto estrutura de tabelas (considere uso de latext), as comparações nos testes, a visualização dos radares chart assim como Critical Difference Diagram.

Vamos refatorar o /analysis e com todas as suas funçoes auxiliares.

Comparações a serem feitos.

- Original vs Backbones (ex: Metr-la vs backbone-metr-la-noise-corrected) pela tabela, mostrando todas as metricas na tabela (MAE,RMSE e WAPE) e uma coluna Delta usando MAE. Essas tabelas devem ser geradas em unico ficheiro por dataset. ex: metr-la.md (todas as tabelas relacionando metrla original e suas versoes simplificadas pelo backbone) e no final do ficheiro, tabela comparando original vs media das versoes de backbone para cada metrica (MAE, RME e WAPE) e depois gera-se a mesmas tabelas na versao .latex

- Mantenha os Testes estatisticos

## Refatoração de de Todo Projeto

### Tarefas:

- Simplificação (deminir verbosidade e complexidade do codigo atual, deixando-o simples e menos verboso)
- Unificação de pipeline e controle pelo parametros com defaults
- - Quero fluxo com intuito de buscar melhores parametros com uso de gridsearchs e e guardadas melhores modelos
- - Fluxo que seleciona já os melhores modelos/parametros para prever/treinar e gerar resultados/relatorios/flots

- Em fluxo de busca de melhor parametro, só será usado dados originais

#### Resumindo a tarefa:

Executo fluxo do pipeline para rodar todos os modelos que selecionei (pelo parametro ou arquivo de configuração) com diferentes combinações de hiperparametros configurados no arquivo de configuração e no final terei cada modelo com seu melhor configuração
Quando seleciono outro fluxo, já vai selecionar rodar modelos que passei pelo parametro com base nas melhores configurações

### Onde Será feito:

Toda essa refatoração será feito criando novo diretorio "refat" deixando old-proj intacto, copiando os necessarios e ou a atualizações necessarias

# Stationdatain-streamlit
# Análise Meteorológica

Este repositório abriga um conjunto de scripts em Python destinados à análise detalhada de dados meteorológicos. Desde o carregamento e pré-processamento dos dados até a implementação de modelos de regressão linear simples, regressão linear múltipla e redes neurais, este projeto oferece uma abordagem abrangente para compreender e prever padrões climáticos.

## Importância

A compreensão dos padrões meteorológicos é essencial em diversas áreas, desde a agricultura até a previsão do tempo. A análise dos dados meteorológicos pode fornecer insights valiosos para tomadas de decisões informadas em várias indústrias, auxiliando, por exemplo, na otimização de culturas, gestão de recursos hídricos e prevenção de eventos climáticos extremos.

## Objetivo

O objetivo principal deste projeto é explorar e analisar dados meteorológicos por meio de técnicas de ciência de dados e aprendizado de máquina. Além disso, busca-se criar modelos preditivos capazes de prever a temperatura externa com base em variáveis meteorológicas específicas. Esta análise pode ser expandida para incluir outras variáveis ou ser adaptada para diferentes conjuntos de dados meteorológicos.

## Aplicações

1. **Agricultura Inteligente:** Prever as condições meteorológicas pode ajudar os agricultores a otimizar o plantio e a colheita, melhorando a produtividade.
2. **Planejamento Urbano:** Compreender padrões climáticos é crucial para o desenvolvimento urbano sustentável e a gestão de recursos.
3. **Prevenção de Desastres:** Modelos preditivos podem contribuir para a prevenção e mitigação de desastres naturais, como enchentes e secas.

## Funcionamento Detalhado

### Carregamento e Limpeza dos Dados

O processo começa com a função `load_data` em `main.py`, responsável por carregar os dados meteorológicos de um arquivo CSV. Em seguida, as funções `convert_to_numeric`, `rename_columns` e `remove_outliers_iqr` são aplicadas para garantir que os dados estejam no formato adequado.

### Análise de Correlação

O código gera um gráfico de correlação usando a biblioteca seaborn, proporcionando uma visão visual das relações entre variáveis meteorológicas selecionadas.

### Regressão Linear Simples

Implementação da regressão linear simples usando scikit-learn. Os dados são divididos em conjuntos de treino e teste, o modelo é treinado e avaliado, e o modelo treinado é persistido usando joblib.

### Regressão Linear Múltipla

Além do treinamento e avaliação do modelo de regressão linear múltipla, a análise inclui a verificação de multicolinearidade usando VIF e a remoção de variáveis com alto VIF.

### Rede Neural

Treinamento e avaliação de uma Rede Neural usando scikit-learn. Os resultados são analisados visualmente através dos resíduos.

### Rede Neural Profunda

Implementação de uma Rede Neural Profunda com arquitetura mais complexa. O modelo é treinado, avaliado e os resíduos são novamente analisados.

## Utilização

1. Certifique-se de ter as dependências instaladas usando `pip install -r requirements.txt`.
2. Execute o arquivo principal `main.py` para iniciar a aplicação Streamlit.
3. Explore os gráficos interativos e métricas de desempenho dos modelos.

## Contribuição

Contribuições são bem-vindas! Se você encontrar problemas ou tiver sugestões de melhorias, sinta-se à vontade para abrir problemas (issues) ou solicitações de pull.

## Licença

Este projeto é licenciado sob a [Met].



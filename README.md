# stationdatain-streamlit
# Análise Meteorológica

Este repositório contém um conjunto de scripts em Python para realizar uma análise abrangente dos dados meteorológicos. O código inclui desde o carregamento e pré-processamento dos dados até a implementação de modelos de regressão linear simples, regressão linear múltipla e redes neurais para previsão de temperatura externa.

## Como Funciona

### Carregamento e Limpeza dos Dados

O processo inicia com a função `load_data` no arquivo `main.py`, que carrega os dados meteorológicos de um arquivo CSV. Em seguida, as funções `convert_to_numeric`, `rename_columns` e `remove_outliers_iqr` são aplicadas para garantir que os dados estejam em formato numérico, renomear colunas conforme conveniência e remover outliers usando a técnica do Intervalo Interquartil (IQR).

### Análise de Correlação

O código gera um gráfico de correlação usando a biblioteca seaborn para visualizar a relação entre as variáveis meteorológicas selecionadas. Esse gráfico é exibido na interface Streamlit.

### Regressão Linear Simples

A regressão linear simples é implementada usando a biblioteca scikit-learn. Os dados são divididos em conjuntos de treino e teste, o modelo é treinado usando o conjunto de treino e avaliado no conjunto de teste. O modelo treinado é persistido usando a biblioteca joblib.

### Regressão Linear Múltipla

Para a regressão linear múltipla, o código verifica a multicolinearidade usando o Valor de Inflação da Variância (VIF) e remove variáveis com alto VIF. O modelo é então treinado e avaliado, e a validação cruzada é aplicada para uma avaliação mais robusta.

### Rede Neural

O código utiliza a biblioteca scikit-learn para treinar uma Rede Neural usando a arquitetura padrão. O modelo é avaliado nos dados de teste, e os resíduos são plotados para análise.

### Rede Neural Profunda

Uma Rede Neural Profunda é implementada com uma arquitetura mais complexa. O modelo é treinado e avaliado nos dados de teste, e os resíduos são novamente plotados para análise.

## Utilização

1. Certifique-se de ter as dependências instaladas usando `pip install -r requirements.txt`.
2. Execute o arquivo principal `main.py` para iniciar a aplicação Streamlit.
3. Explore os gráficos interativos e métricas de desempenho dos modelos.

## Contribuição

Contribuições são bem-vindas! Se você encontrar problemas ou tiver sugestões de melhorias, sinta-se à vontade para abrir problemas (issues) ou solicitações de pull.

## Licença

Este projeto é licenciado sob a [Sua Licença].

---

**Observação:** Substitua "[Sua Licença]" pela licença específica do seu projeto.


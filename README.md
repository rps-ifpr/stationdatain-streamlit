# stationdatain-streamlit
<body>

<p> Analysis of weather station data applied Machine Learning and presented in streamlit
Este código em Python é um script que carrega um conjunto de dados sobre informações meteorológicas em um arquivo csv, e realiza algumas operações de pré-processamento e análise exploratória dos dados. A biblioteca Pandas é utilizada para manipulação e análise dos dados em formato tabular. A biblioteca Seaborn é utilizada para visualização de dados e a biblioteca Matplotlib é utilizada para gerar gráficos e visualizações.</p>

O script realiza as seguintes operações:

Importação das bibliotecas necessárias
Carregamento dos dados em formato csv
Verificação dos tipos de dados das colunas
Conversão de algumas colunas de objetos para numéricos
Renomeação das colunas utilizando um dicionário
Remoção de outliers utilizando o método do intervalo interquartil (IQR)
Remoção de valores nulos
Seleção apenas das colunas numéricas
Criação de um gráfico de correlação dos dados sem outliers
Exibição dos dados em uma tabela
Treinamento de um modelo de Regressão Linear utilizando a biblioteca Scikit-learn (sklearn)
Exibição do gráfico da regressão linear
O script também utiliza a biblioteca Streamlit para criar uma interface simples de usuário, onde é possível visualizar os resultados das operações realizadas.
  
</body>

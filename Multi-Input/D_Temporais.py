import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Carrega os dados do arquivo CSV
dados = pd.read_csv ('dados1975-2015.csv', sep=';', header=0,
                     names=['Ano', 'Mes', 'Chuva', 'Evaporacao', 'Insolacao', 'TempMed', 'UmidRel', 'TempMaxAbs',
                            'TempMaxMed', 'TempMinAbs', 'TempMinMed'])

# Converte as colunas numéricas para float
dados[['Chuva', 'Evaporacao', 'Insolacao', 'TempMed', 'UmidRel', 'TempMaxAbs', 'TempMaxMed', 'TempMinAbs',
       'TempMinMed']] = dados[
    ['Chuva', 'Evaporacao', 'Insolacao', 'TempMed', 'UmidRel', 'TempMaxAbs', 'TempMaxMed', 'TempMinAbs',
     'TempMinMed']].astype (float)

# 1. Análise Descritiva
print (dados.describe ())

# 2. Identificação de Outliers (Boxplot unificado)
plt.figure (figsize=(12, 6))
for i, coluna in enumerate (dados.columns[2:]):
    plt.subplot (3, 3, i + 1)  # Cria subplots para cada variável
    plt.boxplot (dados[coluna], vert=False, patch_artist=True, showmeans=True)
    plt.title (f'Boxplot de {coluna}')
    plt.xlabel (coluna)
plt.tight_layout ()  # Ajusta o layout dos subplots
plt.show ()

# 3. Procurando por Padrões

# Cria um gráfico de dispersão para visualizar a relação entre variáveis
plt.figure (figsize=(8, 6))
plt.scatter (dados['TempMed'], dados['Chuva'], s=20, c='blue', alpha=0.5)
plt.xlabel ('Temperatura Média')
plt.ylabel ('Chuva')
plt.title ('Relação entre Temperatura Média e Chuva')
plt.show ()

# 4. Analisando outliers (opcional)

# Identifica valores que excedem um limite (por exemplo, 3 desvios padrões da média)
for coluna in dados.columns[2:]:
    media = dados[coluna].mean ()
    desvio_padrao = dados[coluna].std ()
    limite_superior = media + 3 * desvio_padrao
    limite_inferior = media - 3 * desvio_padrao
    outliers = dados[coluna][(dados[coluna] > limite_superior) | (dados[coluna] < limite_inferior)]
    print (f'Outliers em {coluna}: {outliers}')

# 5. Analisando padrões (opcional)

# Calcula a correlação entre as variáveis
correlacao = dados.corr ()
print (correlacao)

# Cria um mapa de calor para visualizar a correlação
plt.figure (figsize=(10, 8))
plt.imshow (correlacao, cmap='coolwarm', interpolation='nearest')
plt.colorbar ()
tick_marks = np.arange (len (dados.columns))
plt.xticks (tick_marks, dados.columns, rotation=45)
plt.yticks (tick_marks, dados.columns)
plt.title ('Mapa de Calor de Correlação')
plt.show ()

# -----------------------------------------------------------------------------
# Modelagem de Séries Temporais com LSTM (com Validação Cruzada)
# -----------------------------------------------------------------------------

# Seleciona a variável para modelar
variavel = 'Chuva'  # Escolha a variável que deseja modelar

# Prepara os dados para o modelo LSTM
dados_variavel = dados[[variavel]]

# Normaliza os dados entre 0 e 1
scaler = MinMaxScaler (feature_range=(0, 1))
dados_normalizados = scaler.fit_transform (dados_variavel)

# Cria o modelo LSTM
modelo = Sequential ()
modelo.add (LSTM (50, return_sequences=True, input_shape=(X_treino.shape[1], 1)))
modelo.add (LSTM (50))
modelo.add (Dense (1))
modelo.compile (loss='mean_squared_error', optimizer='adam')

# Define a estratégia de Validação Cruzada
tscv = TimeSeriesSplit (n_splits=5)  # 5 folds para validação cruzada

# Lista para armazenar os resultados do RMSE
rmse_resultados = []

# Loop de Validação Cruzada
for i, (treino_indice, teste_indice) in enumerate (tscv.split (dados_normalizados)):
    print (f'Fold {i + 1}')

    # Divide os dados em treino e teste
    treino_dados = dados_normalizados[treino_indice]
    teste_dados = dados_normalizados[teste_indice]

    # Cria conjuntos de dados para o LSTM
    tempo_lookback = 12  # Número de passos de tempo anteriores para usar como entrada
    X_treino, Y_treino = criar_conjuntos_dados (treino_dados, tempo_lookback)
    X_teste, Y_teste = criar_conjuntos_dados (teste_dados, tempo_lookback)

    # Reshape os dados de entrada para o formato (amostras, passos de tempo, features)
    X_treino = X_treino.reshape (X_treino.shape[0], X_treino.shape[1], 1)
    X_teste = X_teste.reshape (X_teste.shape[0], X_teste.shape[1], 1)

    # Treina o modelo
    modelo.fit (X_treino, Y_treino, epochs=100, batch_size=32, verbose=0)  # Treina sem imprimir o progresso

    # Avalia o modelo nos dados de teste
    previsoes = modelo.predict (X_teste)
    previsoes = scaler.inverse_transform (previsoes)  # Desnormaliza as previsões
    Y_teste = scaler.inverse_transform (Y_teste.reshape (-1, 1))  # Desnormaliza os valores reais

    # Calcula o RMSE para o fold atual
    rmse = mean_squared_error (Y_teste, previsoes, squared=False)
    rmse_resultados.append (rmse)
    print (f'RMSE para o Fold {i + 1}: {rmse}')

# Plota a função de perda durante o treinamento (opcional)
plt.figure (figsize=(10, 6))
plt.plot (modelo.history.history['loss'], label='Função de Perda')
plt.xlabel ('Época')
plt.ylabel ('Perda')
plt.title ('Função de Perda Durante o Treinamento')
plt.legend ()
plt.show ()

# Calcula o RMSE médio para todos os folds
rmse_medio = np.mean (rmse_resultados)
print (f'RMSE Médio: {rmse_medio}')

# Plota as previsões vs. valores reais para o último fold
plt.figure (figsize=(12, 6))
plt.plot (Y_teste, label='Real')
plt.plot (previsoes, label='Previsões')
plt.legend ()
plt.title (f'Previsões vs. Reais ({variavel}) - Último Fold')
plt.show ()
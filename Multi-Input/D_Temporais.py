import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # Adiciona StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Adiciona Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error  # Adiciona outras métricas
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.optimizers import Adam  # Adiciona Adam
from tensorflow.keras.regularizers import l1, l2  # Adiciona regularização L1 e L2
import numpy as np  # Adiciona a importação do NumPy

# Carrega os dados
dados = pd.read_csv('dados1975-2015.csv', sep=';', header=0,
                     names=['Ano', 'Mes', 'Chuva', 'Evaporacao', 'Insolacao', 'TempMed', 'UmidRel', 'TempMaxAbs',
                            'TempMaxMed', 'TempMinAbs', 'TempMinMed'])

# Converte para float
dados[['Chuva', 'Evaporacao', 'Insolacao', 'TempMed', 'UmidRel', 'TempMaxAbs', 'TempMaxMed', 'TempMinAbs',
       'TempMinMed']] = dados[
    ['Chuva', 'Evaporacao', 'Insolacao', 'TempMed', 'UmidRel', 'TempMaxAbs', 'TempMaxMed', 'TempMinAbs',
     'TempMinMed']].astype(float)

# 1. Análise Descritiva
print(dados.describe())

# 2. Identificação de Outliers (Boxplot unificado)
plt.figure(figsize=(12, 6))
for i, coluna in enumerate(dados.columns[2:]):
    plt.subplot(3, 3, i+1)  # Cria subplots para cada variável
    plt.boxplot(dados[coluna], vert=False, patch_artist=True, showmeans=True)
    plt.title(f'Boxplot de {coluna}')
    plt.xlabel(coluna)
plt.tight_layout()  # Ajusta o layout dos subplots
plt.show()

# 3. Procurando por Padrões

# Cria um gráfico de dispersão para visualizar a relação entre variáveis
plt.figure(figsize=(8, 6))
plt.scatter(dados['TempMed'], dados['Chuva'], s=20, c='blue', alpha=0.5)
plt.xlabel('Temperatura Média')
plt.ylabel('Chuva')
plt.title('Relação entre Temperatura Média e Chuva')
plt.show()

# 4. Analisando outliers (opcional)

# Identifica valores que excedem um limite (por exemplo, 3 desvios padrões da média)
for coluna in dados.columns[2:]:
    media = dados[coluna].mean()
    desvio_padrao = dados[coluna].std()
    limite_superior = media + 3 * desvio_padrao
    limite_inferior = media - 3 * desvio_padrao
    outliers = dados[coluna][(dados[coluna] > limite_superior) | (dados[coluna] < limite_inferior)]
    print(f'Outliers em {coluna}: {outliers}')

# 5. Analisando padrões (opcional)

# Calcula a correlação entre as variáveis
correlacao = dados.corr()
print(correlacao)

# Cria um mapa de calor para visualizar a correlação
plt.figure(figsize=(10, 8))
plt.imshow(correlacao, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(dados.columns))
plt.xticks(tick_marks, dados.columns, rotation=45)
plt.yticks(tick_marks, dados.columns)
plt.title('Mapa de Calor de Correlação')
plt.show()

# -----------------------------------------------------------------------------
# Modelagem de Séries Temporais com LSTM (com Validação Cruzada)
# -----------------------------------------------------------------------------

# Seleciona a variável
variavel = 'Chuva'

# Prepara os dados
dados_variavel = dados[[variavel]]

# Normaliza os dados (experimente StandardScaler)
scaler = MinMaxScaler(feature_range=(0, 1))  # Ou StandardScaler
dados_normalizados = scaler.fit_transform(dados_variavel)

# Define a estratégia de Validação Cruzada (experimente TimeSeriesSplit)
tscv = TimeSeriesSplit(n_splits=5)

# Lista para armazenar os resultados do RMSE
rmse_resultados = []
mae_resultados = []
mape_resultados = []

# Função para criar conjuntos de dados para o LSTM
def criar_conjuntos_dados(dados, tempo_lookback):
    X, Y = [], []
    for i in range(len(dados)-tempo_lookback-1):
        a = dados[i:(i+tempo_lookback), 0]
        X.append(a)
        Y.append(dados[i + tempo_lookback, 0])
    return np.array(X), np.array(Y)

# Loop de Validação Cruzada
for i, (treino_indice, teste_indice) in enumerate(tscv.split(dados_normalizados)):
    print(f'Fold {i + 1}')

    # Divide os dados
    treino_dados = dados_normalizados[treino_indice]
    teste_dados = dados_normalizados[teste_indice]

    # Cria conjuntos de dados (experimente diferentes valores para tempo_lookback)
    tempo_lookback = 12
    X_treino, Y_treino = criar_conjuntos_dados(treino_dados, tempo_lookback)
    X_teste, Y_teste = criar_conjuntos_dados(teste_dados, tempo_lookback)

    # Reshape
    X_treino = X_treino.reshape(X_treino.shape[0], X_treino.shape[1], 1)
    X_teste = X_teste.reshape(X_teste.shape[0], X_teste.shape[1], 1)

    # Cria o modelo (experimente diferentes configurações)
    modelo = Sequential()
    modelo.add(LSTM(50, return_sequences=True, input_shape=(X_treino.shape[1], 1),
                    recurrent_regularizer=l2(0.01)))  # Adiciona regularização L2
    modelo.add(Dropout(0.2))  # Adiciona dropout
    modelo.add(LSTM(50, recurrent_regularizer=l2(0.01)))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(1))
    modelo.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))  # Experimente Adam

    # Treina o modelo (experimente diferentes valores para epochs e batch_size)
    historico = modelo.fit(X_treino, Y_treino, epochs=100, batch_size=32, verbose=0, validation_data=(X_teste, Y_teste))

    # Avalia o modelo
    previsoes = modelo.predict(X_teste)
    previsoes = scaler.inverse_transform(previsoes)
    Y_teste = scaler.inverse_transform(Y_teste.reshape(-1, 1))

    # Calcula as métricas
    rmse = mean_squared_error(Y_teste, previsoes, squared=False)
    mae = mean_absolute_error(Y_teste, previsoes)
    mape = mean_absolute_percentage_error(Y_teste, previsoes)

    rmse_resultados.append(rmse)
    mae_resultados.append(mae)
    mape_resultados.append(mape)
    print(f'RMSE para o Fold {i + 1}: {rmse}')
    print(f'MAE para o Fold {i + 1}: {mae}')
    print(f'MAPE para o Fold {i + 1}: {mape}')

# Visualizações (expanda com gráficos de erros, previsões de vários passos)
# Plota a função de perda durante o treinamento (mesmo código)
plt.figure(figsize=(10, 6))
plt.plot(historico.history['loss'], label='Perda no Conjunto de Treinamento')
plt.plot(historico.history['val_loss'], label='Perda no Conjunto de Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.title('Curva de Aprendizagem')
plt.legend()
plt.show()

# Calcula as métricas médias
rmse_medio = np.mean(rmse_resultados)
mae_medio = np.mean(mae_resultados)
mape_medio = np.mean(mape_resultados)

print(f'RMSE Médio: {rmse_medio}')
print(f'MAE Médio: {mae_medio}')
print(f'MAPE Médio: {mape_medio}')

# Plota as previsões vs. valores reais (mesmo código)
plt.figure(figsize=(12, 6))
plt.plot(Y_teste, label='Real')
plt.plot(previsoes, label='Previsões')
plt.legend()
plt.title(f'Previsões vs. Reais ({variavel}) - Último Fold')
plt.show()

# Gráfico de Dispersão (para todas as combinações de variáveis)
plt.figure(figsize=(10, 6))
plt.scatter(dados['TempMed'], dados['Chuva'], s=20, c='blue', alpha=0.5, label='Chuva')
plt.scatter(dados['TempMed'], dados['Evaporacao'], s=20, c='green', alpha=0.5, label='Evaporacao')
plt.xlabel('Temperatura Média')
plt.ylabel('Valores')
plt.title('Relação entre Temperatura Média e Chuva e Evaporacao')
plt.legend()
plt.show()

# Gráfico de Erros
plt.figure(figsize=(10, 6))
plt.plot(Y_teste - previsoes, label='Erros de Previsão')
plt.xlabel('Índice de Tempo')
plt.ylabel('Erro')
plt.title('Erros de Previsão')
plt.legend()
plt.show()

# Previsões de Múltiplos Passos (opcional)
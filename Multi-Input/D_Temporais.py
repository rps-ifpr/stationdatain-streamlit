import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # Adiciona StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Adiciona Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error  # Adiciona outras métricas
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from tensorflow.keras.optimizers import Adam  # Adiciona Adam
from tensorflow.keras.regularizers import l1, l2  # Adiciona regularização L1 e L2
import numpy as np  # Adiciona a importação do NumPy
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor  # Importação corrigida

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

# 2. Identificação e Tratamento de Outliers (Boxplot unificado)
for coluna in dados.columns[2:]:
    # Calcula limites superiores e inferiores
    limite_superior = dados[coluna].mean() + 3 * dados[coluna].std()
    limite_inferior = dados[coluna].mean() - 3 * dados[coluna].std()

    # Substitui outliers pela média
    dados[coluna] = np.clip(dados[coluna], limite_inferior, limite_superior)

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

# 4. Analisando padrões (opcional)

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
# Modelagem de Séries Temporais com LSTM (com Validação Cruzada e Otimização)
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

# Função para criar conjuntos de dados para o LSTM
def criar_conjuntos_dados(dados, tempo_lookback):
    X, Y = [], []
    for i in range(len(dados)-tempo_lookback-1):
        a = dados[i:(i+tempo_lookback), 0]
        X.append(a)
        Y.append(dados[i + tempo_lookback, 0])
    return np.array(X), np.array(Y)

# Cria o modelo LSTM
def criar_modelo(n_unidades_lstm=50, dropout=0.2, learning_rate=0.001):
    modelo = Sequential()
    modelo.add(LSTM(n_unidades_lstm, return_sequences=True, input_shape=(X_treino.shape[1], 1),
                    recurrent_regularizer=l2(0.01)))  # Adiciona regularização L2
    modelo.add(Dropout(dropout))  # Adiciona dropout
    modelo.add(LSTM(n_unidades_lstm, recurrent_regularizer=l2(0.01)))
    modelo.add(Dropout(dropout))
    modelo.add(Dense(1))
    modelo.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))  # Experimente Adam
    return modelo

# Cria o objeto KerasRegressor
modelo = KerasRegressor(build_fn=criar_modelo, epochs=100, batch_size=32, verbose=0)

# Define os parâmetros para otimização
parâmetros = {'n_unidades_lstm': [25, 50, 100],
              'dropout': [0.1, 0.2, 0.3],
              'learning_rate': [0.001, 0.01, 0.1]}

# Cria o objeto GridSearchCV
grid_search = GridSearchCV(estimator=modelo, param_grid=parâmetros, scoring='neg_mean_squared_error', cv=tscv)

# Loop de Validação Cruzada com Otimização
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

    # Realiza o Grid Search
    grid_search.fit(X_treino, Y_treino)

    # Avalia o modelo com os melhores hiperparâmetros
    previsoes = grid_search.best_estimator_.predict(X_teste)
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

    # Imprime os melhores hiperparâmetros
    print(f'Melhores hiperparâmetros para o Fold {i+1}: {grid_search.best_params_}')

# Imprime os melhores hiperparâmetros encontrados no Grid Search
print(f'Melhores hiperparâmetros encontrados: {grid_search.best_params_}')

# Avalia o modelo com os melhores hiperparâmetros em todo o conjunto de dados
# (Depois que o Grid Search for executado em todos os folds)
modelo_final = grid_search.best_estimator_
previsoes_final = modelo_final.predict(X_teste)
previsoes_final = scaler.inverse_transform(previsoes_final)
Y_teste = scaler.inverse_transform(Y_teste.reshape(-1, 1))

# Calcula as métricas
rmse_final = mean_squared_error(Y_teste, previsoes_final, squared=False)
mae_final = mean_absolute_error(Y_teste, previsoes_final)
mape_final = mean_absolute_percentage_error(Y_teste, previsoes_final)

print(f'RMSE Final: {rmse_final}')
print(f'MAE Final: {mae_final}')
print(f'MAPE Final: {mape_final}')

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

# Plota as previsões vs. valores reais (mesmo código)
plt.figure(figsize=(12, 6))
plt.plot(Y_teste, label='Real')
plt.plot(previsoes_final, label='Previsões')
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
plt.plot(Y_teste - previsoes_final, label='Erros de Previsão')
plt.xlabel('Índice de Tempo')
plt.ylabel('Erro')
plt.title('Erros de Previsão')
plt.legend()
plt.show()

# -----------------------------------------------------------------------------
# Previsões de Múltiplos Passos
# -----------------------------------------------------------------------------

# Define a variável
variavel = 'Chuva'

# Prepara os dados
dados_variavel = dados[[variavel]]

# Normaliza os dados
scaler = MinMaxScaler(feature_range=(0, 1))
dados_normalizados = scaler.fit_transform(dados_variavel)

# Cria o modelo LSTM com os melhores hiperparâmetros
modelo = criar_modelo(n_unidades_lstm=grid_search.best_params_['n_unidades_lstm'],
                    dropout=grid_search.best_params_['dropout'],
                    learning_rate=grid_search.best_params_['learning_rate'])

# Treina o modelo com todos os dados (sem validação cruzada)
modelo.fit(X_treino, Y_treino, epochs=100, batch_size=32, verbose=0)

# Previsões de Múltiplos Passos
n_passos_futuro = 12  # Número de passos para prever
previsoes_multiplas = []

# Cria uma cópia dos dados de teste para realizar previsões sequenciais
dados_teste_copia = dados_normalizados[len(treino_dados)-tempo_lookback:]

for i in range(n_passos_futuro):
    # Cria o conjunto de dados de entrada para a previsão
    X_entrada = dados_teste_copia[i:i+tempo_lookback].reshape(1, tempo_lookback, 1)

    # Faz a previsão
    previsao = modelo.predict(X_entrada)

    # Adiciona a previsão à lista
    previsoes_multiplas.append(previsao[0,0])

    # Atualiza o conjunto de dados de teste com a previsão para o próximo passo
    dados_teste_copia = np.concatenate((dados_teste_copia, previsao), axis=0)

# Desnormaliza as previsões de múltiplos passos
previsoes_multiplas = scaler.inverse_transform(np.array(previsoes_multiplas).reshape(-1, 1))

# Plota as previsões de múltiplos passos
plt.figure(figsize=(12, 6))
plt.plot(Y_teste, label='Real')
plt.plot(np.arange(len(Y_teste), len(Y_teste)+n_passos_futuro), previsoes_multiplas, label='Previsões')
plt.legend()
plt.title(f'Previsões de Múltiplos Passos ({variavel})')
plt.show()
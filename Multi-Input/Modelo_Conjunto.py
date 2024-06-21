import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Carregue os dados combinados
dados_combinados = pd.read_csv('dados_combinados.csv')

# Defina as variáveis de entrada e saída
variaveis_entrada = ['satelite_solo', 'satelite_agua', 'satelite_vegetacao', 'umidade_solo',
                    'condutividade_eletrica', 'temperatura_solo', 'estagio_cultura',
                    'precipitacao_previsao', 'chuva_historica']
variavel_saida = 'irrigacao_previsao'

# Pré-processamento dos dados
scaler = MinMaxScaler()
dados_combinados[variaveis_entrada] = scaler.fit_transform(dados_combinados[variaveis_entrada])
dados_combinados[variavel_saida] = scaler.fit_transform(dados_combinados[[variavel_saida]])

# Crie o modelo LSTM
modelo_conjunto = Sequential()
modelo_conjunto.add(LSTM(50, return_sequences=True, input_shape=(9, 1), recurrent_regularizer=l2(0.01)))  # 9 é o número de features de entrada
modelo_conjunto.add(Dropout(0.2))
modelo_conjunto.add(LSTM(50, recurrent_regularizer=l2(0.01)))
modelo_conjunto.add(Dropout(0.2))
modelo_conjunto.add(Dense(1))
modelo_conjunto.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

# Treinamento com validação cruzada
tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []
mae_scores = []
mape_scores = []

for train_index, test_index in tscv.split(dados_combinados):
    X_treinamento, X_teste = dados_combinados[variaveis_entrada].iloc[train_index], dados_combinados[variaveis_entrada].iloc[test_index]
    y_treinamento, y_teste = dados_combinados[variavel_saida].iloc[train_index], dados_combinados[variavel_saida].iloc[test_index]

    # Treine o modelo
    modelo_conjunto.fit(X_treinamento, y_treinamento, epochs=100, batch_size=32, verbose=1)

    # Faça previsões no conjunto de teste
    y_previsao = modelo_conjunto.predict(X_teste)

    # Avalie o desempenho do modelo
    rmse = np.sqrt(mean_squared_error(y_teste, y_previsao))
    mae = mean_absolute_error(y_teste, y_previsao)
    mape = mean_absolute_percentage_error(y_teste, y_previsao)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)

# Imprima as métricas de desempenho
print(f'RMSE: {np.mean(rmse_scores)}')
print(f'MAE: {np.mean(mae_scores)}')
print(f'MAPE: {np.mean(mape_scores)}')

# Salve o modelo treinado
modelo_conjunto.save('modelo_conjunto.h5')
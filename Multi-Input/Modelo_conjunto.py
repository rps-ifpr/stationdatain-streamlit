import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

# Carregue os dados combinados
dados_combinados = pd.read_csv('dados_combinados.csv')

# Defina as variáveis de entrada e saída
variaveis_entrada = ['satelite_solo', 'satelite_agua', 'satelite_vegetacao', 'umidade_solo',
                    'condutividade_eletrica', 'temperatura_solo', 'estagio_cultura',
                    'precipitacao_previsao', 'chuva_historica']
variavel_saida = 'irrigacao_previsao'

# Crie um objeto LabelEncoder
label_encoder = LabelEncoder()

# Converta a coluna 'estagio_cultura' para valores numéricos
dados_combinados['estagio_cultura'] = label_encoder.fit_transform(dados_combinados['estagio_cultura'])

# Crie um objeto MinMaxScaler
scaler = MinMaxScaler()

# Normalize as variáveis de entrada
dados_combinados[variaveis_entrada] = scaler.fit_transform(dados_combinados[variaveis_entrada])

# Normalize a variável de saída
dados_combinados[variavel_saida] = scaler.fit_transform(dados_combinados[[variavel_saida]])

# Crie o modelo LSTM
modelo_conjunto = Sequential()
modelo_conjunto.add(LSTM(50, return_sequences=True, input_shape=(9, 1), recurrent_regularizer=l2(0.01)))
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

# Lista para armazenar a perda de treinamento e validação em cada fold
loss_train = []
loss_val = []

for train_index, test_index in tscv.split(dados_combinados):
    X_treinamento, X_teste = dados_combinados[variaveis_entrada].iloc[train_index], dados_combinados[variaveis_entrada].iloc[test_index]
    y_treinamento, y_teste = dados_combinados[variavel_saida].iloc[train_index], dados_combinados[variavel_saida].iloc[test_index]

    # Treine o modelo
    history = modelo_conjunto.fit(X_treinamento, y_treinamento, epochs=100, batch_size=32, validation_data=(X_teste, y_teste), verbose=0)

    # Faça previsões no conjunto de teste
    y_previsao = modelo_conjunto.predict(X_teste)

    # Avalie o desempenho do modelo
    rmse = np.sqrt(mean_squared_error(y_teste, y_previsao))
    mae = mean_absolute_error(y_teste, y_previsao)
    mape = mean_absolute_percentage_error(y_teste, y_previsao)

    rmse_scores.append(rmse)
    mae_scores.append(mae)
    mape_scores.append(mape)

    # Armazene a perda de treinamento e validação para cada fold
    loss_train.append(history.history['loss'])
    loss_val.append(history.history['val_loss'])

# Imprima as métricas de desempenho
print(f'RMSE: {np.mean(rmse_scores)}')
print(f'MAE: {np.mean(mae_scores)}')
print(f'MAPE: {np.mean(mape_scores)}')

# Salve o modelo treinado
modelo_conjunto.save('modelo_conjunto.h5')

# Plota a curva de aprendizagem
plt.figure(figsize=(10, 6))
plt.plot(np.mean(loss_train, axis=0), label='Perda de Treinamento')
plt.plot(np.mean(loss_val, axis=0), label='Perda de Validação')
plt.title('Curva de Aprendizagem do Modelo LSTM')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.show()

# Plota a perda ao longo das épocas para cada fold
plt.figure(figsize=(10, 6))
for i in range(len(loss_train)):
    plt.plot(loss_train[i], label=f'Fold {i+1} - Treinamento')
    plt.plot(loss_val[i], label=f'Fold {i+1} - Validação')

plt.title('Perda ao Longo das Épocas (Validação Cruzada)')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.show()
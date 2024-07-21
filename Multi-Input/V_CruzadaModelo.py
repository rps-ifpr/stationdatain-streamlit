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

# Plota a curva de aprendizagem de cada fold em um único gráfico
fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8))
fig.suptitle('Learning Curve (Cross Validation) for each Fold', fontsize=16)

for i, (train_index, test_index) in enumerate(tscv.split(dados_combinados)):
    # Separe os dados
    X_treinamento, X_teste = dados_combinados[variaveis_entrada].iloc[train_index], \
    dados_combinados[variaveis_entrada].iloc[test_index]
    y_treinamento, y_teste = dados_combinados[variavel_saida].iloc[train_index], dados_combinados[variavel_saida].iloc[
        test_index]

    # Treine o modelo (utilizando verbose=0 para evitar mensagens de treino na tela)
    history = modelo_conjunto.fit(X_treinamento, y_treinamento, epochs=100, batch_size=32,
                                  validation_data=(X_teste, y_teste), verbose=0)

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

    # Plota a perda do fold atual
    axes[i].plot(history.history['loss'], label=f'Training {i + 1}')
    axes[i].plot(history.history['val_loss'], label=f'Validation {i + 1}')
    axes[i].set_ylabel('Loss')
    axes[i].legend()

# Define o label do eixo X (comparti lhado) e o título para o último fold
axes[-1].set_xlabel('Epoch')
axes[-2].set_title('k-Fold Cross Validation - Loss in each Epoch', fontsize=16)

plt.tight_layout()  # Ajusta o layout para evitar sobreposição
plt.show()

# Imprima as métricas de desempenho
print(f'RMSE: {np.mean(rmse_scores)}')
print(f'MAE: {np.mean(mae_scores)}')
print(f'MAPE: {np.mean(mape_scores)}')

# Salve o modelo treinado
modelo_conjunto.save('modelo_conjunto.h5')
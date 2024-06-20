import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Carregar os dados
data = pd.read_csv("dados1975-2015.csv", sep=",")  # Assumindo que o separador é ','

# Criar uma coluna de data a partir de 'Ano' e 'Mês'
data['data'] = pd.to_datetime(data[['Ano', 'Mês']])
data.set_index('data', inplace=True) # Define a coluna de data como índice

# Selecionar as variáveis relevantes
features = ["TempMed", "UmidRel", "Insolação"]
target = "Evaporação"

# Pré-processamento:
# 1. Normalizar os dados
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# 2. Criar conjuntos de treinamento e teste
train_data = data[:-36]  # Usar os últimos 36 meses para teste
test_data = data[-36:]

# 3. Criar conjuntos de dados para o modelo RNN (usando janelas deslizantes)
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, target])
    return np.array(X), np.array(Y)

look_back = 12  # Número de meses a considerar como entrada para prever o próximo mês
X_train, y_train = create_dataset(train_data.values, look_back)
X_test, y_test = create_dataset(test_data.values, look_back)

# 4. Definir o modelo RNN (LSTM neste exemplo)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Define a função de programação da taxa de aprendizado
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.9

# Crie um objeto LearningRateScheduler
lr_scheduler = LearningRateScheduler(scheduler)

# 5. Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), lr_scheduler],
                    verbose=1)

# 6. Avaliar o modelo
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Erro Médio Quadrático (MSE): {loss}")

# 7. Fazer previsões
y_pred = model.predict(X_test)

# 8. Inverter a normalização para obter as previsões na escala original
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 9. Plotar as previsões vs. valores reais
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Real')
plt.plot(y_pred, label='Previsão')
plt.title('Previsões de Evapotranspiração')
plt.xlabel('Tempo')
plt.ylabel('Evapotranspiração (mm/dia)')
plt.legend()
plt.show()

# Plotar a curva de aprendizagem (perda e acurácia ao longo das épocas)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda de Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia de Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.tight_layout()
plt.show()
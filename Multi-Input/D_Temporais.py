import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
data = pd.read_csv("dados1975-2015.csv", parse_dates=["data"], index_col="data")

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

# 5. Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

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
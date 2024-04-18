import matplotlib.pyplot as plt
import numpy as np

# Definindo as características e suas respectivas importâncias
features = ['Satélite', 'Série Temporal', 'Meteorológico', 'Umidade Solo', 'Infor. Culturas']
importances = [0.35, 0.65, 0.50, 0.75, 0.90]
colors = ['blue', 'orange', 'green', 'red', 'purple']

# Criando o gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(features, importances, color=colors)

# Adicionando título e rótulos
plt.title('Importância das Características no Modelo de Predição')
plt.ylabel('Importância')

# Exibindo o gráfico
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Simulando um conjunto de dados com valores aleatórios para as variáveis de interesse
np.random.seed(0)
data = np.random.rand(200, 5)  # 100 observações e 5 características

# Simulando nomes de colunas como as características do seu modelo
columns = ['Satélite', 'Série Temporal', 'Meteorológico', 'Umidade do Solo', 'Infor Culturas']

# Calculando a matriz de correlação
correlation_matrix = np.corrcoef(data, rowvar=False)

# Criando o heatmap de correlação
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, xticklabels=columns, yticklabels=columns, cmap='coolwarm', center=0)
plt.title('')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate
from keras.datasets import boston_housing

# Carregar um conjunto de dados exemplo
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# Preparar dados tabulares e de séries temporais
X_tabular_train = X_train[:, :5]  # Primeiras 5 características como dados tabulares
X_series_train = X_train[:, 5:].reshape(-1, 8, 1)  # Restante como série temporal
X_tabular_test = X_test[:, :5]
X_series_test = X_test[:, 5:].reshape(-1, 8, 1)

# Modelo para dados tabulares
tabular_input = Input(shape=(5,))
tabular_model = Dense(64, activation='relu')(tabular_input)
tabular_model = Dense(32, activation='relu')(tabular_model)

# Modelo para séries temporais
series_input = Input(shape=(8, 1))
series_model = Conv1D(32, 3, activation='relu')(series_input)
series_model = MaxPooling1D(2)(series_model)
series_model = LSTM(32)(series_model)

# Combinar os modelos
combined = concatenate([tabular_model, series_model])
output = Dense(1)(combined)
model = Model(inputs=[tabular_input, series_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Treinar o modelo
history = model.fit([X_tabular_train, X_series_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

# Avaliar o modelo
model.evaluate([X_tabular_test, X_series_test], y_test)

# Gráfico de Desempenho
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss (training data)')
plt.plot(history.history['val_loss'], label='Loss (validation data)')
plt.title('Loss Evolution')
plt.ylabel('Loss')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='MAE (training data)')
plt.plot(history.history['val_mae'], label='MAE (validation data)')
plt.title('MAE Evolution')
plt.ylabel('MAE')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()



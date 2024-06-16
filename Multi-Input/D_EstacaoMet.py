import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados (assumindo que os dados estão em um arquivo CSV chamado 'estacao_meteorologica.csv')
data = pd.read_csv('estacao_meteorologica.csv', sep=';', parse_dates=['Time'])  # Ajuste o separador se necessário

# Converter as colunas para numéricas
data['Outdoor Temperature(°C)'] = pd.to_numeric(data['Outdoor Temperature(°C)'], errors='coerce')
data['Outdoor Humidity(%)'] = pd.to_numeric(data['Outdoor Humidity(%)'], errors='coerce')
data['Wind Speed(km/h)'] = pd.to_numeric(data['Wind Speed(km/h)'], errors='coerce')
data['Gust(km/h)'] = pd.to_numeric(data['Gust(km/h)'], errors='coerce')
data['DewPoint(°C)'] = pd.to_numeric(data['DewPoint(°C)'], errors='coerce')
data['WindChill(°C)'] = pd.to_numeric(data['WindChill(°C)'], errors='coerce')

# Remover linhas com valores faltantes
data.dropna(inplace=True)

# Definir as variáveis explicativas (X) e a variável resposta (y)
X = data[['Indoor Temperature(°C)', 'Indoor Humidity(%)', 'Relative Pressure(mmHg)', 'Absolute Pressure(mmHg)', 'Wind Speed(km/h)', 'Gust(km/h)', 'DewPoint(°C)', 'WindChill(°C)', 'Hour Rainfall(mm)', '24 Hour Rainfall(mm)', 'Week Rainfall(mm)', 'Month Rainfall(mm)', 'Total Rainfall(mm)']]
y = data['Outdoor Temperature(°C)']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados (opcional, mas recomendado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar o modelo de regressão linear múltipla
model = LinearRegression()

# Gerar a curva de aprendizagem
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10)
)

# Calcular as médias e desvios padrão dos scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plotar a curva de aprendizagem
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Treinamento')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Validação')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='green')
plt.xlabel("Tamanho do Conjunto de Treinamento")
plt.ylabel("MSE")
plt.title("Curva de Aprendizagem do Modelo de Regressão")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir os resultados
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

# Obter os coeficientes do modelo
coefficients = model.coef_
intercept = model.intercept_

# Imprimir os coeficientes
print("Coeficientes:", coefficients)
print("Intercepto:", intercept)
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import numpy as np  # Importando numpy para gerar os tamanhos dos conjuntos de treinamento

# Carregar os dados
data = pd.read_csv("dados_sensor.csv", parse_dates=["data"])

# 2. Verificar outliers (usar boxplots)
data.boxplot(column=["umidade_solo", "condutividade_eletrica", "temperatura_solo"])
plt.show()

# 3. Lidar com outliers (opcional - remover outliers com base no boxplot)
threshold = 1.5
for col in ["umidade_solo", "condutividade_eletrica", "temperatura_solo"]:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[col] < (Q1 - threshold * IQR)) | (data[col] > (Q3 + threshold * IQR)))]

# 3. Verificar valores faltantes
print("Valores faltantes:", data.isnull().sum())

# 3. Lidar com valores faltantes (remover linhas com valores faltantes)
data.dropna(inplace=True)

# Preparar os dados para o modelo
# 1. Separar variáveis explicativas (X) e variável resposta (y)
X = data[["temperatura_solo"]]  # Usar apenas temperatura como variável explicativa
y = data["umidade_solo"]

# 2. Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalizar os dados (opcional, mas geralmente recomendado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar o modelo de regressão linear
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
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='Validation')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='green')
plt.xlabel("Training Set Size")
plt.ylabel("MSE")
plt.title("Regression Model Learning Curve")
plt.legend(loc="best")
plt.grid(True)
plt.show()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular métricas de avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir os resultados
print("Erro Quadrático Médio:", mse)
print("Coeficiente de Determinação (R²):", r2)

# Salvar o modelo treinado
filename = 'modelo_sensor.sav'
pickle.dump(model, open(filename, 'wb')) # Aqui está a linha que salva o modelo

# --- Gráficos para Análise do Treinamento ---

# 1. Gráfico de Dispersão (Temperatura vs. Umidade)
plt.figure(figsize=(8, 6))
plt.scatter(data["temperatura_solo"], data["umidade_solo"], s=20, c='blue', alpha=0.7)
plt.xlabel("Temperatura do Solo (°C)")
plt.ylabel("Umidade do Solo (%)")
plt.title("Relação entre Temperatura e Umidade do Solo")
plt.grid(True)
plt.show()

# 2. Histograma dos Resíduos (Modelo de Umidade)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel("Resíduos")
plt.ylabel("Frequência")
plt.title("Histograma dos Resíduos do Modelo de Umidade")
plt.show()

# 3. Gráfico de Previsões vs. Valores Reais (Modelo de Umidade)
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y_test, s=20, c='green', alpha=0.7)
plt.xlabel("Previsões de Umidade do Solo (%)")
plt.ylabel("Valores Reais de Umidade do Solo (%)")
plt.title("Previsões vs. Valores Reais do Modelo de Umidade")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Linha Ideal')
plt.legend()
plt.grid(True)
plt.show()
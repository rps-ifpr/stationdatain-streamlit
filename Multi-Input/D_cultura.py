from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt


# Carregar os dados
dados_cultura = pd.read_csv('dados_cultura.csv')

# Separar as features (X) e a variável alvo (y)
X = dados_cultura[['estagio_cultura', 'tipo_solo', 'tipo_irrigacao', 'cultivar']]
y = dados_cultura['demanda_agua']

# Converter variáveis categóricas para numéricas (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de árvore de decisão
modelo_arvore = DecisionTreeRegressor(random_state=42)

# Treinar o modelo
modelo_arvore.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = modelo_arvore.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Médio Quadrático (MSE): {mse}")

# Visualizar a árvore (opcional)
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(modelo_arvore, feature_names=X.columns, filled=True)
plt.show()
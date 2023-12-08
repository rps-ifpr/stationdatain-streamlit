import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Simulando dados para o exemplo
data = {
    'external_temp': [25, 28, 22, 30, 26, 27],
    'absolute_pressure': [1015, 1010, 1012, 1017, 1013, 1011],
    'relative_pressure': [1012, 1008, 1010, 1015, 1011, 1009]
}

df = pd.DataFrame(data)

# Treinando modelo de Regressão Linear para exemplo
X_train, X_test, y_train, y_test = train_test_split(
    df[['absolute_pressure']],
    df['external_temp'],
    test_size=0.2,
    random_state=42
)

reg = LinearRegression()
reg.fit(X_train, y_train)

# Gráfico de Dispersão com Linha de Regressão
fig, ax = plt.subplots()
ax.scatter(df['absolute_pressure'], df['external_temp'], label='Observado')
ax.plot(df['absolute_pressure'], reg.predict(df[['absolute_pressure']]), color='red', label='Regressão Linear')
ax.set_xlabel('Absolute Pressure')
ax.set_ylabel('External Temperature')
ax.legend()
st.pyplot(fig)

# Visualização da Árvore de Decisão
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=df['absolute_pressure'], y=df['external_temp'], label='Observado', ax=ax)
sns.lineplot(x=df['absolute_pressure'], y=tree_model.predict(df[['absolute_pressure']]), color='red', label='Árvore de Decisão', ax=ax)
ax.set_xlabel('Absolute Pressure')
ax.set_ylabel('External Temperature')
ax.legend()
st.pyplot(fig)

# Comparação entre Observado e Previsto
predictions_reg = reg.predict(df[['absolute_pressure']])
predictions_tree = tree_model.predict(df[['absolute_pressure']])

df_predictions = pd.DataFrame({
    'Absolute Pressure': df['absolute_pressure'],
    'Observado': df['external_temp'],
    'Reg. Linear': predictions_reg,
    'Árvore de Decisão': predictions_tree
})

st.write("Comparação entre Observado e Previsto")
st.line_chart(df_predictions.set_index('Absolute Pressure'))

# Gráfico de Resíduos para Regressão Linear
residuals_reg = df['external_temp'] - predictions_reg
fig, ax = plt.subplots()
ax.scatter(df['absolute_pressure'], residuals_reg)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Absolute Pressure')
ax.set_ylabel('Resíduos (Observado - Previsto)')
st.pyplot(fig)

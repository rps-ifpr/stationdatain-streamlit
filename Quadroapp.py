import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import statsmodels.api as sm
import joblib  # Certifique-se de instalar a biblioteca: pip install joblib
import plotly.express as px

# Função para carregar dados
def load_data(file_path, delimiter=';'):
    return pd.read_csv(file_path, delimiter=delimiter)

# Função para converter variáveis para numéricas
def convert_to_numeric(df, columns):
    for coluna in columns:
        df[coluna] = pd.to_numeric(df[coluna], errors='coerce')

# Função para renomear colunas
def rename_columns(df, column_mapping):
    df.rename(columns=column_mapping, inplace=True)

# Função para remover outliers usando IQR
def remove_outliers_iqr(df, columns, multiplier=1.5):
    for coluna in columns:
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        filtro = (df[coluna] >= Q1 - multiplier * IQR) & (df[coluna] <= Q3 + multiplier * IQR)
        df = df.loc[filtro]
    return df

# Função para normalizar variáveis
def normalize_variables(df, columns, scaler):
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    return df_normalized

# Função para aplicar transformação logarítmica
def apply_log_transform(df, column):
    df[column] = np.log1p(df[column])
    return df

# Função para treinar modelo de regressão linear
def train_linear_regression_model(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    return reg

# Função para calcular o VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [sm.OLS(X[col].values, sm.add_constant(X.drop(col, axis=1))).fit().rsquared for col in X.columns]
    return vif_data

# Carregando os dados
df = load_data('dados.csv')

# Convertendo variáveis numéricas que estão como objetos
colunas_para_converter = ['Outdoor Temperature(°C)', 'Outdoor Humidity(%)', 'Wind Speed(km/h)',
                          'Gust(km/h)', 'DewPoint(°C)', 'WindChill(°C)']
convert_to_numeric(df, colunas_para_converter)

# Renomeando as colunas
rename_columns(df, {
    'n': 'id',
    'Time': 'date',
    'Interval': 'intervalo',
    'Indoor Temperature(°C)': 'internal_temp',
    'Indoor Humidity(%)': 'internal_humidity',
    'Outdoor Temperature(°C)': 'external_temp',
    'Outdoor Humidity(%)': 'external_humidity',
    'Relative Pressure(mmHg)': 'relative_pressure',
    'Absolute Pressure(mmHg)': 'absolute_pressure',
    'Wind Speed(km/h)': 'wind_speed',
    'Gust(km/h)': 'gust_wind',
    'Wind Direction': 'wind_direction',
    'DewPoint(°C)': 'dew_point',
    'WindChill(°C)': 'thermal_sensation',
    'Hour Rainfall(mm)': 'rain_time',
    '24 Hour Rainfall(mm)': 'rain_24h',
    'Week Rainfall(mm)': 'rain_week',
    'Month Rainfall(mm)': 'rain_month',
    'Total Rainfall(mm)': 'total_rain'
})

# Removendo outliers
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
df_clean = remove_outliers_iqr(df, colunas_numericas)

# Removendo valores nulos
df_clean.dropna(inplace=True)

# Selecionando colunas para análise
colunas_analise = ['external_temp', 'external_humidity', 'wind_speed', 'gust_wind', 'dew_point', 'thermal_sensation', 'absolute_pressure']

# Título da aplicação Streamlit
st.title("Estação Meteorológica IFPR-Campus Capanema - Estudo de Caso 2")

# Criando dataframe de correlação
corr = df_clean[colunas_analise].corr()

# Plotando o gráfico de correlação
st.write("Gráfico de correlação dos dados sem outliers")
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Verificando a multicolinearidade usando o VIF
X = df_clean[colunas_analise]
vif_data = calculate_vif(X)

# Removendo variáveis com alto VIF (por exemplo, VIF > 5)
high_vif_variables = vif_data[vif_data['VIF'] > 5]['Variable'].tolist()
df_clean = df_clean.drop(columns=high_vif_variables)

# Plotando o gráfico de barras para o VIF
st.write("Gráfico de VIF para cada variável:")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Variable', y='VIF', data=vif_data, palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

# Selecionando colunas para análise após remover variáveis com alto VIF
colunas_analise = ['external_temp', 'absolute_pressure']

# Dividindo os dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(df_clean[colunas_analise], df_clean['external_temp'], test_size=0.2, random_state=42)

# Treinando o modelo de Regressão Linear nos dados de treino
reg = train_linear_regression_model(X_train, y_train)

# Persistindo o modelo treinado
joblib.dump(reg, 'modelo_regressao_linear.joblib')

# Carregando o modelo treinado
loaded_model = joblib.load('modelo_regressao_linear.joblib')

# Avaliando o modelo nos dados de teste
score = loaded_model.score(X_test, y_test)
st.write(f'Acurácia do modelo nos dados de teste: {score}')

# Exibindo o coeficiente e o intercepto do modelo carregado
st.write('Coeficiente do modelo carregado:', loaded_model.coef_[0])
st.write('Intercepto do modelo carregado:', loaded_model.intercept_)

# Plotando o gráfico com a regressão linear usando o modelo RL-simples
st.write("Gráfico com a regressão linear usando o modelo carregado")
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='absolute_pressure', y='external_temp', data=df_clean, ax=ax, line_kws={'color': 'red'})
st.pyplot(fig)

######## REGRESSÃO LINEAR MÚLTIPLA ########

# Selecionando colunas para análise de regressão linear múltipla
colunas_analise_multipla = ['external_humidity', 'wind_speed', 'gust_wind', 'dew_point', 'thermal_sensation', 'absolute_pressure']

# Dividindo os dados para treino e teste para regressão linear múltipla
X_multipla_train, X_multipla_test, y_multipla_train, y_multipla_test = train_test_split(df_clean[colunas_analise_multipla], df_clean['external_temp'], test_size=0.2, random_state=42)

# Normalizando as variáveis para a regressão linear múltipla
scaler_multipla = StandardScaler()
X_multipla_train_normalized = normalize_variables(X_multipla_train, colunas_analise_multipla, scaler_multipla)
X_multipla_test_normalized = normalize_variables(X_multipla_test, colunas_analise_multipla, scaler_multipla)

# Treinando o modelo de Regressão Linear Múltipla nos dados de treino
reg_multipla = train_linear_regression_model(X_multipla_train_normalized, y_multipla_train)

# Persistindo o modelo treinado de regressão linear múltipla
joblib.dump(reg_multipla, 'modelo_regressao_linear_multipla.joblib')

# Carregando o modelo treinado de regressão linear múltipla
loaded_model_multipla = joblib.load('modelo_regressao_linear_multipla.joblib')

# Plotando o gráfico com a regressão linear múltipla usando o modelo carregado
st.write("Gráfico com a regressão linear múltipla usando o modelo RL-multipla")

# Criando DataFrame para comparar valores reais e previstos
df_resultados_multipla = pd.DataFrame({
    'Valores Reais': y_multipla_test,
    'Valores Previstos': loaded_model_multipla.predict(X_multipla_test_normalized)
})

# Gráfico de dispersão
fig_multipla = px.scatter(df_resultados_multipla, x='Valores Previstos', y='Valores Reais', template="plotly",
                          title="Regressão Linear Múltipla - Valores Reais vs. Previstos", color_discrete_sequence=['blue'])

# Linha de tendência
fig_multipla.add_trace(px.line(x=df_resultados_multipla['Valores Previstos'],
                               y=loaded_model_multipla.predict(X_multipla_test_normalized),
                               line_shape='linear').data[0])

# Adicionando identidade (linha 45 graus) para referência
fig_multipla.add_shape(type='line', x0=df_resultados_multipla['Valores Previstos'].min(),
                       x1=df_resultados_multipla['Valores Previstos'].max(),
                       y0=df_resultados_multipla['Valores Previstos'].min(),
                       y1=df_resultados_multipla['Valores Previstos'].max(),
                       line=dict(color='red', dash='dash'))

st.plotly_chart(fig_multipla)

######################################Variavel de maior correlação

# Calcular a matriz de correlação
correlation_matrix_multipla = df_clean[colunas_analise_multipla + ['external_temp']].corr()

# Selecionar variáveis independentes com correlação significativa
correlation_threshold_multipla = 0.5  # Ajuste conforme necessário
significant_features_multipla = correlation_matrix_multipla[abs(correlation_matrix_multipla['external_temp']) > correlation_threshold_multipla].index

# Usar apenas variáveis independentes com correlação significativa
colunas_analise_multipla = significant_features_multipla[:-1]  # Excluindo a variável dependente 'external_temp'

# Dividindo os dados para treino e teste para regressão linear múltipla
X_multipla_train, X_multipla_test, y_multipla_train, y_multipla_test = train_test_split(
    df_clean[colunas_analise_multipla], df_clean['external_temp'], test_size=0.2, random_state=42
)

# Normalizando as variáveis para a regressão linear múltipla
scaler_multipla = StandardScaler()
X_multipla_train_normalized = normalize_variables(X_multipla_train, colunas_analise_multipla, scaler_multipla)
X_multipla_test_normalized = normalize_variables(X_multipla_test, colunas_analise_multipla, scaler_multipla)

# Treinando o modelo de Regressão Linear Múltipla nos dados de treino
reg_multipla = train_linear_regression_model(X_multipla_train_normalized, y_multipla_train)

# Persistindo o modelo treinado de regressão linear múltipla
joblib.dump(reg_multipla, 'modelo_regressao_linear_multipla2.joblib')

# Carregando o modelo treinado de regressão linear múltipla
loaded_model_multipla = joblib.load('modelo_regressao_linear_multipla2.joblib')

# Selecionar variáveis independentes com correlação significativa
correlation_threshold_multipla = 0.5  # Ajuste conforme necessário
significant_features_multipla = correlation_matrix_multipla[abs(correlation_matrix_multipla['external_temp']) > correlation_threshold_multipla].index

# Usar apenas variáveis independentes com correlação significativa
colunas_analise_multipla = significant_features_multipla[:-1]  # Excluindo a variável dependente 'external_temp'

# Imprimir as variáveis usadas
st.write("Variáveis independentes usadas na regressão linear múltipla:")
st.write(colunas_analise_multipla)

# Dividindo os dados para treino e teste para regressão linear múltipla
X_multipla_train, X_multipla_test, y_multipla_train, y_multipla_test = train_test_split(
    df_clean[colunas_analise_multipla], df_clean['external_temp'], test_size=0.2, random_state=42
)


# Plotando o gráfico com a regressão linear múltipla usando o modelo carregado
st.write("Gráfico com a regressão linear múltipla usando o modelo carregado")
fig_multipla = px.scatter(df_resultados_multipla, x='Valores Previstos', y='Valores Reais', template="plotly",
                          title="Regressão Linear Múltipla - Valores Reais vs. Previstos", color_discrete_sequence=['blue'])

# Linha de tendência
fig_multipla.add_trace(px.line(x=df_resultados_multipla['Valores Previstos'],
                               y=loaded_model_multipla.predict(X_multipla_test_normalized),
                               line_shape='linear').data[0])

# Adicionando identidade (linha 45 graus) para referência
fig_multipla.add_shape(type='line', x0=df_resultados_multipla['Valores Previstos'].min(),
                       x1=df_resultados_multipla['Valores Previstos'].max(),
                       y0=df_resultados_multipla['Valores Previstos'].min(),
                       y1=df_resultados_multipla['Valores Previstos'].max(),
                       line=dict(color='red', dash='dash'))

st.plotly_chart(fig_multipla)




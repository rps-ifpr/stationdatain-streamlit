import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

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

# Definindo as variáveis explicativas e a variável alvo
explanatory_variables = ['absolute_pressure']
target_variable = 'external_temp'

# Treinando o modelo de Regressão Linear
reg = train_linear_regression_model(df_clean[explanatory_variables], df_clean[target_variable])

# Exibindo o coeficiente e o intercepto
st.write('Coeficiente:', reg.coef_[0])
st.write('Intercepto:', reg.intercept_)

# Plotando o gráfico com a regressão linear
st.write("Gráfico com a regressão linear")
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='absolute_pressure', y='external_temp', data=df_clean, ax=ax, line_kws={'color': 'red'})
st.pyplot(fig)


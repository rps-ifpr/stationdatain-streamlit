import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense

# Importando os dados
df = pd.read_csv('dados.csv', delimiter=';')

# Verificando tipo de dados
tipos = df.dtypes
st.write(tipos)

# Converter variáveis numéricas que estão como objetos
colunas_numericas = ['Outdoor Temperature(°C)', 'Outdoor Humidity(%)', 'Wind Speed(km/h)',
                     'Gust(km/h)', 'DewPoint(°C)', 'WindChill(°C)']
df[colunas_numericas] = df[colunas_numericas].apply(pd.to_numeric, errors='coerce')

# Criando o dicionário com os novos nomes
df_novo = {
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
}

# Renomeando as colunas
df.rename(columns=df_novo, inplace=True)
df.info()

# Definindo a função de remoção de outliers
def remove_outliers_iqr(df, multiplier=1.5):
    df_clean = df.copy()
    for column in df_clean.select_dtypes(include='number').columns:
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    return df_clean

# Removendo os outliers
df_clean = remove_outliers_iqr(df)

# Removendo valores nulos
df_clean = df_clean.dropna()

# Selecionando apenas colunas numéricas
num_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns

# Removendo colunas desnecessárias
num_cols = num_cols.drop(['id', 'intervalo', 'rain_time', 'rain_week', 'rain_month', 'rain_24h'])




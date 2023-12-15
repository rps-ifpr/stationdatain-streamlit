import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Função para carregar e processar dados
def load_and_process_data(file_path, delimiter=';'):
    df = pd.read_csv(file_path, delimiter=delimiter)

    # Convertendo variáveis para numéricas
    colunas_para_converter = ['Outdoor Temperature(°C)', 'Outdoor Humidity(%)', 'Wind Speed(km/h)',
                              'Gust(km/h)', 'DewPoint(°C)', 'WindChill(°C)']
    df[colunas_para_converter] = df[colunas_para_converter].apply(pd.to_numeric, errors='coerce')

    # Renomeando colunas
    column_mapping = {
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
    df.rename(columns=column_mapping, inplace=True)

    # Removendo outliers
    colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for coluna in colunas_numericas:
        Q1, Q3 = df[coluna].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        filtro = (df[coluna] >= Q1 - 1.5 * IQR) & (df[coluna] <= Q3 + 1.5 * IQR)
        df = df.loc[filtro]

    # Removendo valores nulos
    df.dropna(inplace=True)

    return df

# Título da aplicação Streamlit
st.title("Estação Meteorológica IFPR-Campus Capanema - Estudo de Caso 3")

# Carregando e processando os dados
df = load_and_process_data('dados.csv')

# Selecionando colunas para análise
colunas_analise = ['external_temp', 'external_humidity', 'wind_speed', 'gust_wind', 'dew_point', 'thermal_sensation', 'absolute_pressure']

# Criando dataframe de correlação
corr = df[colunas_analise].corr()

# Plotando o gráfico de correlação
st.write("Gráfico de correlação dos dados sem outliers")
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
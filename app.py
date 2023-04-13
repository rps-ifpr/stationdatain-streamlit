import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import streamlit as st

# importando os dados
df= pd.read_csv ('dados.csv', delimiter=';')

# Verificando tipo de dados
tipos = df.dtypes
print(tipos)

# Converter variáveis numéricas que estão como objetos
df['Outdoor Temperature(°C)'] = pd.to_numeric(df['Outdoor Temperature(°C)'], errors='coerce')
df['Outdoor Humidity(%)'] = pd.to_numeric(df['Outdoor Humidity(%)'], errors='coerce')
df['Wind Speed(km/h)'] = pd.to_numeric(df['Wind Speed(km/h)'], errors='coerce')
df['Gust(km/h)'] = pd.to_numeric(df['Gust(km/h)'], errors='coerce')
df['DewPoint(°C)'] = pd.to_numeric(df['DewPoint(°C)'], errors='coerce')
df['WindChill(°C)'] = pd.to_numeric(df['WindChill(°C)'], errors='coerce')

# criando o dicionário com os novos nomes
df_novo = {'n': 'id',
             'Time': 'date',
             'Interval': 'intervalo',
             'Indoor Temperature(°C)': 'internal_temp',
             'Indoor Humidity(%)': 'internal_humidity',
             'Outdoor Temperature(°C)': 'external_temp',
             'Outdoor Humidity(%)': 'external_humidity',
             'Relative Pressure(mmHg)': 'relative_pressure',
             'Absolute Pressure(mmHg)': 'absolute_pressure',
             'Wind Speed(km/h)': 'wind speed',
             'Gust(km/h)': 'gust_wind',
             'Wind Direction': 'wind_direction',
             'DewPoint(°C)': 'dew point',
             'WindChill(°C)': 'Thermal sensation',
             'Hour Rainfall(mm)': 'rain_time',
             '24 Hour Rainfall(mm)': 'rain_24h',
             'Week Rainfall(mm)': 'rain_week',
             'Month Rainfall(mm)': 'rain_month',
             'Total Rainfall(mm)': 'total_rain'}
df.info()

# renomeando as colunas
df.rename(columns=df_novo, inplace=True)

# Definindo a função de remoção de outliers
def remove_outliers_iqr(df, multiplier=1.5):
    # Cria uma cópia do dataframe original
    df_clean = df.copy ()
    # Itera sobre todas as colunas numéricas
    for column in df_clean.select_dtypes (include='number').columns:
        # Calcula os limites inferior e superior usando o IQR
        Q1 = df_clean[column].quantile (0.25)
        Q3 = df_clean[column].quantile (0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        # Remove os outliers
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    return df_clean
# Removendo os outliers
df_clean = remove_outliers_iqr(df)

# Removendo valores nulos
df_clean = df_clean.dropna()

# selecionando apenas colunas numéricas
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# removendo colunas desnecessárias
num_cols = num_cols.drop(['id', 'intervalo','rain_time','rain_week','rain_month','rain_24h'])

st.title("Estação Meteorológica IFPR-Campus Capanema")

# criando dataframe de correlação
corr = df_clean[num_cols].corr()
# Plotando o gráfico de correlação
st.write(f"<p style='font-size: 25px'>{'Gráfico de correlação dos dados sem outliers'}</p>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

def correlation_plot():
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Exibindo os dados
st.write(f"<p style='font-size: 25px'>{'Aqui são apresentados os dados em tabela da estação Local:'}</p>", unsafe_allow_html=True)

st.write(df_clean.head())

# Definindo as variáveis explicativas e a variável alvo
explanatory_variables = ['absolute_pressure']
target_variable = ['external_temp']
# Treinando o modelo Regressão Linear
reg = LinearRegression()
reg.fit(df_clean[explanatory_variables], df_clean[target_variable])
st.write(f"<p style='font-size: 25px'>{'Gráfico com a regressão linear 1'}</p>", unsafe_allow_html=True)
# Exibindo o coeficiente e o intercepto
st.write(f'Coeficiente: {reg.coef_[0]:}')
st.write(f'Intercepto: {reg.intercept_:}')
# Exibindo o gráfico com a regressão linear 1
fig, ax = plt.subplots()
ax.scatter(df_clean[explanatory_variables], df_clean[target_variable])
ax.plot(df_clean[explanatory_variables], reg.predict(df_clean[explanatory_variables]), color='red')
ax.set_xlabel('Absolute Pressure')
ax.set_ylabel('External Temperature')
st.pyplot(fig)
# Definindo as variáveis explicativas e a variável alvo
explanatory_variables = ['relative_pressure']
target_variable = ['external_temp']
# Treinando o modelo Regressão Linear
reg = LinearRegression()
reg.fit(df_clean[explanatory_variables], df_clean[target_variable])
st.write(f"<p style='font-size: 25px'>{'Graph with linear regression 2'}</p>",
         unsafe_allow_html=True)
# Exibindo o coeficiente e o intercepto
st.write(f'Coeficiente: {reg.coef_[0]:}')
st.write(f'Intercepto: {reg.intercept_:}')
# Exibindo o gráfico com a regressão linear 2
fig, ax = plt.subplots()
ax.scatter(df_clean[explanatory_variables], df_clean[target_variable])
ax.plot(df_clean[explanatory_variables], reg.predict(df_clean[explanatory_variables]), 
        color='red')
ax.set_xlabel('Relative Pressure')
ax.set_ylabel('External Temperature')
st.pyplot(fig)





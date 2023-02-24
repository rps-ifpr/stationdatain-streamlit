import pandas as pd
import seaborn as sns
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
             'Time': 'data',
             'Interval': 'intervalo',
             'Indoor Temperature(°C)': 'temp_interna',
             'Indoor Humidity(%)': 'umidade_interna',
             'Outdoor Temperature(°C)': 'temp_externa',
             'Outdoor Humidity(%)': 'umidade_externa',
             'Relative Pressure(mmHg)': 'pressao_relativa',
             'Absolute Pressure(mmHg)': 'pressao_absoluta',
             'Wind Speed(km/h)': 'velocidade_vento',
             'Gust(km/h)': 'rajada_vento',
             'Wind Direction': 'direcao_vento',
             'DewPoint(°C)': 'ponto_orvalho',
             'WindChill(°C)': 'sensacao_termica',
             'Hour Rainfall(mm)': 'chuva_hora',
             '24 Hour Rainfall(mm)': 'chuva_24h',
             'Week Rainfall(mm)': 'chuva_semana',
             'Month Rainfall(mm)': 'chuva_mes',
             'Total Rainfall(mm)': 'chuva_total'}
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

# selecionando apenas colunas numéricas
num_cols = df.select_dtypes(include=['float64', 'int64']).columns

# removendo colunas 'id' e 'Intervalo'
num_cols = num_cols.drop(['id', 'intervalo','chuva_hora','chuva_semana','chuva_mes','chuva_24h'])

st.title("Estação Meteorológica IFPR-Campus Capanema")
st.write("Aqui estão os dados da estação Local:")

# criando dataframe de correlação 1
corr_df = df[num_cols].corr()
# Plotando o gráfico de correlação
st.write("Gráfico de correlação dos dados")
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
def correlation_plot():
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(corr_df(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# criando dataframe de correlação 2
corr = df_clean[num_cols].corr()
# Plotando o gráfico de correlação
st.write("Gráfico de correlação dos dados sem outliers")
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

def correlation_plot():
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Filtro na sidebar para selecionar as linhas de temperatura
st.sidebar.header('Filtro de Temperatura')
temp_columns = ['Indoor Temperature(°C)', 'Outdoor Temperature(°C)', 'DewPoint(°C)', 'WindChill(°C)']
selected_columns = st.sidebar.multiselect('Selecione as linhas de temperatura', temp_columns)
if selected_columns:
    new_df = df[selected_columns]
    title = 'Correlação entre as variáveis de temperatura selecionadas'
    correlation_plot(new_df, title)



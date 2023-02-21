import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# importando os dados
df= pd.read_csv ('dados.csv', delimiter=';')

st.title("Dados da estação meteorológica")
st.write("Aqui estão os dados de estação Local:")

first_10_rows = df.head(10)
st.write(first_10_rows)

# Converter a coluna de data/hora para o tipo datetime
df['Date/Time'] = pd.to_datetime(df['Time'])

# Criar uma nova coluna com a data arredondada para o dia mais próximo
df['Day'] = df['Date/Time'].dt.floor('D')

# Agrupar os dados por dia e calcular a média da temperatura interna
daily_avg_temp = df.groupby('Day')['Indoor Temperature(°C)'].mean()

st.write(daily_avg_temp)





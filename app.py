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

# Criar um menu suspenso para selecionar o período
periodo = st.sidebar.selectbox('Período', ['Últimos 7 dias', 'Últimos 30 dias', 'Todos os dados'])

# Selecionar o subconjunto apropriado de dados
if periodo == 'Últimos 7 dias':
    data = daily_avg_temp.tail(7)
elif periodo == 'Últimos 30 dias':
    data = daily_avg_temp.tail(30)
else:
    data = daily_avg_temp

# Criar um gráfico de linha com a temperatura média por dia
fig, ax = plt.subplots()
ax.plot(daily_avg_temp.index, daily_avg_temp.values)
ax.set_xlabel('Data')
ax.set_ylabel('Temperatura Média (°C)')
ax.set_title('Temperatura Média por Dia')
ax.grid(True)

# Exibir o gráfico no Streamlit
st.pyplot(fig)





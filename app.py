import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# importando os dados
df= pd.read_csv ('dados.csv', delimiter=';')

st.title("Dados da estação meteorológica")
st.write("Aqui estão os dados de estação Local:")

#first_10_rows = df.head(10)
#st.write(first_10_rows)

# Converter a coluna de data/hora para o tipo datetime
df['Date'] = pd.to_datetime(df['Time'])

# Criar uma nova coluna com a data arredondada para o dia mais próximo
df['Day'] = df['Date'].dt.floor('D')

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

# Exibir o gráfico no Streamlit
st.pyplot(fig)

# Criar um slider na barra lateral
temp_min = int(df["Indoor Temperature(°C)"].min())
temp_max = int(df["Indoor Temperature(°C)"].max())
temp_default = int(df["Indoor Temperature(°C)"].mean())

temperature = st.sidebar.slider("Selecione a temperatura interna (°C)", temp_min, temp_max, temp_default)

# Filtrar dados pelo valor do controle deslizante
filtered_data = df[df["Indoor Temperature(°C)"] == temperature]

# Criar um gráfico de linhas com a temperatura interna ao longo do tempo
fig, ax = plt.subplots()
ax.plot(filtered_data["Time"], filtered_data["Indoor Temperature(°C)"])
ax.set_xlabel("Tempo")
ax.set_ylabel("Temperatura interna (°C)")
ax.set_title("Temperatura interna ao longo do tempo")

# Exibir gráfico com o Streamlit
st.pyplot(fig)




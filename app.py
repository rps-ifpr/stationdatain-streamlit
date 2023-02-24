import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# importando os dados
df= pd.read_csv ('dados.csv', delimiter=';')

st.title("Dados da estação meteorológica")
st.write("Aqui estão os dados de estação Local:")

# Criando a matriz de correlação
corr = df.corr()

# Plotando o gráfico de correlação
st.write("Gráfico de correlação dos dados")
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

def correlation_plot():
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
    st.pyplot(fig)

# Filtro na sidebar para selecionar as linhas de temperatura
st.sidebar.header('Filtro de Temperatura')
temp_columns = ['Indoor Temperature(°C)', 'Outdoor Temperature(°C)', 'DewPoint(°C)', 'WindChill(°C)']
selected_columns = st.sidebar.multiselect('Selecione as linhas de temperatura', temp_columns)

if selected_columns:
    new_df = df[selected_columns]
    title = 'Correlação entre as variáveis de temperatura selecionadas'
    correlation_plot(new_df, title)

# Plota o heatmap com todas as variáveis
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlação entre as variáveis selecionadas')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
st.pyplot()


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Função para pré-processar os dados
def preprocessar_dados(df):
    # 1. Limpeza de Dados
    df['umidade_solo'] = df['umidade_solo'][df['umidade_solo'] >= 0]  # Remover valores inválidos de umidade
    df['condutividade_eletrica'] = df['condutividade_eletrica'][df['condutividade_eletrica'] >= 0]  # Remover valores inválidos de condutividade
    df['temperatura_solo'] = df['temperatura_solo'][df['temperatura_solo'] >= 0]  # Remover valores inválidos de temperatura

    # 2. Normalização (Min-Max Scaling Global)
    df['umidade_solo'] = (df['umidade_solo'] - df['umidade_solo'].min()) / (df['umidade_solo'].max() - df['umidade_solo'].min())
    df['condutividade_eletrica'] = (df['condutividade_eletrica'] - df['condutividade_eletrica'].min()) / (df['condutividade_eletrica'].max() - df['condutividade_eletrica'].min())
    df['temperatura_solo'] = (df['temperatura_solo'] - df['temperatura_solo'].min()) / (df['temperatura_solo'].max() - df['temperatura_solo'].min())

    return df

# Interface Streamlit
st.title("Pré-processamento de Dados de Sensores de Solo")

# Ler dados do arquivo CSV
df = pd.read_csv('dados_sensor.csv', parse_dates=['data'])  # Lê o arquivo CSV e converte a coluna 'data' para o tipo Timestamp

# Pré-processar os dados
df_preprocessado = preprocessar_dados(df.copy())

# Criar gráficos para comparação
st.subheader("Dados Pré-processados")
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Umidade do Solo
axs[0].plot(df_preprocessado['data'], df_preprocessado['umidade_solo'], label="Pré-processado")
axs[0].set_ylabel("Umidade do Solo (%)")
axs[0].legend()

# Condutividade Elétrica
axs[1].plot(df_preprocessado['data'], df_preprocessado['condutividade_eletrica'], label="Pré-processado")
axs[1].set_ylabel("Condutividade Elétrica (mS/cm)")
axs[1].legend()

# Temperatura do Solo
axs[2].plot(df_preprocessado['data'], df_preprocessado['temperatura_solo'], label="Pré-processado")
axs[2].set_ylabel("Temperatura do Solo (°C)")
axs[2].legend()

# Ajustar espaçamento entre os subplots
plt.tight_layout()

# Mostrar o gráfico
st.pyplot(fig)

# Análise Mensal - Agrupamento por Mês e cálculo da média
st.subheader("Análise Mensal dos Dados")
df['mes'] = df['data'].dt.month
df_preprocessado['mes'] = df_preprocessado['data'].dt.month  # Adicione a coluna 'mes' ao df_preprocessado

# Criar Gráficos Mensais
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Criar uma lista para os meses
meses = ['Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez', 'Jan', 'Fev', 'Mar', 'Abr', 'Mai']

# Umidade do Solo
for mes in range(1, 13):
    df_mensal_bruto = df[df['mes'] == mes].groupby('data').mean()
    df_mensal_preprocessado = df_preprocessado[df_preprocessado['mes'] == mes].groupby('data').mean()
    axs[0].plot(df_mensal_bruto.index, df_mensal_bruto['umidade_solo'], label=f"Bruto - {meses[mes - 1]}")
    axs[0].plot(df_mensal_preprocessado.index, df_mensal_preprocessado['umidade_solo'], label=f"Pré-processado - {meses[mes - 1]}")

# Condutividade Elétrica
for mes in range(1, 13):
    df_mensal_bruto = df[df['mes'] == mes].groupby('data').mean()
    df_mensal_preprocessado = df_preprocessado[df_preprocessado['mes'] == mes].groupby('data').mean()
    axs[1].plot(df_mensal_bruto.index, df_mensal_bruto['condutividade_eletrica'], label=f"Bruto - {meses[mes - 1]}")
    axs[1].plot(df_mensal_preprocessado.index, df_mensal_preprocessado['condutividade_eletrica'], label=f"Pré-processado - {meses[mes - 1]}")

# Temperatura do Solo
for mes in range(1, 13):
    df_mensal_bruto = df[df['mes'] == mes].groupby('data').mean()
    df_mensal_preprocessado = df_preprocessado[df_preprocessado['mes'] == mes].groupby('data').mean()
    axs[2].plot(df_mensal_bruto.index, df_mensal_bruto['temperatura_solo'], label=f"Bruto - {meses[mes - 1]}")
    axs[2].plot(df_mensal_preprocessado.index, df_mensal_preprocessado['temperatura_solo'], label=f"Pré-processado - {meses[mes - 1]}")


# Ajustar espaçamento entre os subplots
plt.tight_layout()

# Mostrar o gráfico
st.pyplot(fig)

# Classificação dos Dados
st.subheader("Classificação dos Dados")
from sklearn.cluster import KMeans

# Definir o número de clusters (classes)
n_clusters = 3  # Ajustar conforme necessário

# Criar o modelo KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Ajustar o modelo aos dados pré-processados
kmeans.fit(df_preprocessado[['umidade_solo', 'condutividade_eletrica', 'temperatura_solo']])

# Obter as labels (classes) de cada ponto de dados
labels = kmeans.labels_

# Adicionar as labels ao DataFrame
df_preprocessado['classe'] = labels

# Criar um gráfico para visualizar os clusters
st.subheader("Visualização dos Clusters")
plt.figure(figsize=(10, 6))

# Plotar os clusters usando diferentes cores para cada classe
plt.scatter(df_preprocessado['umidade_solo'], df_preprocessado['condutividade_eletrica'], c=df_preprocessado['classe'], cmap='viridis')
plt.xlabel("Umidade do Solo (%)")
plt.ylabel("Condutividade Elétrica (mS/cm)")
plt.title("Clusters de Dados de Sensores de Solo")

st.pyplot(plt)

# Criar tabela com estatísticas descritivas de cada cluster
st.subheader("Estatísticas Descritivas dos Clusters")
df_cluster_stats = df_preprocessado.groupby('classe').agg(media_umidade=('umidade_solo', 'mean'),
                                                 media_condutividade=('condutividade_eletrica', 'mean'),
                                                 media_temperatura=('temperatura_solo', 'mean'))

st.dataframe(df_cluster_stats)
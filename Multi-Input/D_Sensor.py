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
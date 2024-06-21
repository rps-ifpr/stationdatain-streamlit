import pandas as pd
import numpy as np
import random

# Define o número de linhas para cada conjunto de dados
num_linhas = 1000

# Função de dados dentro de um intervalo
def gerar_dados_aleatorios(minimo, maximo, num_linhas):
  return np.random.uniform(minimo, maximo, num_linhas)

# Gera os dados para cada modelo
dados_satelite = pd.DataFrame({
    'satelite_solo': gerar_dados_aleatorios(0, 1, num_linhas),
    'satelite_agua': gerar_dados_aleatorios(0, 1, num_linhas),
    'satelite_vegetacao': gerar_dados_aleatorios(0, 1, num_linhas),
})

dados_sensores = pd.DataFrame({
    'umidade_solo': gerar_dados_aleatorios(0, 100, num_linhas),
    'condutividade_eletrica': gerar_dados_aleatorios(0, 10, num_linhas),
    'temperatura_solo': gerar_dados_aleatorios(10, 35, num_linhas),
})

dados_cultura = pd.DataFrame({
    'estagio_cultura': [random.choice(['Vegetativo', 'Floração', 'Colheita']) for _ in range(num_linhas)],
})

dados_estacao = pd.DataFrame({
    'precipitacao_previsao': gerar_dados_aleatorios(0, 50, num_linhas),
})

dados_temporais = pd.DataFrame({
    'chuva_historica': gerar_dados_aleatorios(0, 100, num_linhas),
})

# Combina os dados usando pd.concat (eixo 1 para combinar as colunas)
dados_combinados = pd.concat([dados_satelite, dados_sensores, dados_cultura, dados_estacao, dados_temporais], axis=1)

# Salva os dados combinados em um novo arquivo CSV
dados_combinados.to_csv('dados_combinados.csv', index=False)

print("Dados combinados salvos em dados_combinados.csv")
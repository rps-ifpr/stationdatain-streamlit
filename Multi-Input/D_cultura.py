import pandas as pd
import numpy as np
import random

# Define o número de linhas para o conjunto de dados
num_linhas = 1000

# Define os estágios de desenvolvimento do lúpulo
estagios = ['Vegetativo', 'Floração', 'Colheita']

# Define os tipos de solo
tipos_solo = ['Franco-arenoso', 'Argiloso', 'Areia']

# Define os tipos de irrigação
tipos_irrigacao = ['Aspersão', 'Gotejamento']

# Define as cultivares de lúpulo (opcional)
cultivares = ['Cascade', 'Citra', 'Chinook']

# Função para gerar dados aleatórios dentro de um intervalo
def gerar_dados_aleatorios(minimo, maximo, num_linhas):
    return np.random.uniform(minimo, maximo, num_linhas)

# Cria um DataFrame com dados aleatórios
dados_cultura = pd.DataFrame({
    'estagio_cultura': [random.choice(estagios) for _ in range(num_linhas)],
    'tipo_solo': [random.choice(tipos_solo) for _ in range(num_linhas)],
    'tipo_irrigacao': [random.choice(tipos_irrigacao) for _ in range(num_linhas)],
    'demanda_agua': gerar_dados_aleatorios(0, 100, num_linhas)  # Gera demanda de água aleatória (ajuste o intervalo conforme necessário)
})

# Adiciona uma coluna para cultivares (opcional)
if cultivares:
    dados_cultura['cultivar'] = [random.choice(cultivares) for _ in range(num_linhas)]

# Salva o DataFrame como um arquivo CSV
dados_cultura.to_csv('dados_cultura.csv', index=False)

print("Dados da cultura do lúpulo salvos em dados_cultura.csv")
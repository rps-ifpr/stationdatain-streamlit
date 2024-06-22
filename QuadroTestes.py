import pandas as pd
import numpy as np
import random

# Define o número de linhas para cada conjunto de dados
num_linhas = 1000

# Define os estágios de desenvolvimento do lúpulo
estagios = ['Vegetativo', 'Floração', 'Colheita']

# Função para gerar dados aleatórios dentro de um intervalo
def gerar_dados_aleatorios(minimo, maximo, num_linhas):
    return np.random.uniform(minimo, maximo, num_linhas)

# Gera os dados de previsão para cada modelo (valores entre 0 e 1)
previsoes_satelite = pd.DataFrame({
    'satelite_solo': gerar_dados_aleatorios(0, 1, num_linhas),
    'satelite_agua': gerar_dados_aleatorios(0, 1, num_linhas),
    'satelite_vegetacao': gerar_dados_aleatorios(0, 1, num_linhas),
    'estagio_cultura': [random.choice(estagios) for _ in range(num_linhas)]
})

previsoes_sensores = pd.DataFrame({
    'umidade_solo': gerar_dados_aleatorios(0, 1, num_linhas),
    'condutividade_eletrica': gerar_dados_aleatorios(0, 1, num_linhas),
    'temperatura_solo': gerar_dados_aleatorios(0, 1, num_linhas),
    'estagio_cultura': [random.choice(estagios) for _ in range(num_linhas)]
})

previsoes_cultura = pd.DataFrame({
    'cultura_demanda': gerar_dados_aleatorios(0, 1, num_linhas),
    'estagio_cultura': [random.choice(estagios) for _ in range(num_linhas)]
})

previsoes_estacao = pd.DataFrame({
    'estacao_precipitacao': gerar_dados_aleatorios(0, 1, num_linhas),
    'estagio_cultura': [random.choice(estagios) for _ in range(num_linhas)]
})

previsoes_temporais = pd.DataFrame({
    'temporais_chuva': gerar_dados_aleatorios(0, 1, num_linhas),
    'estagio_cultura': [random.choice(estagios) for _ in range(num_linhas)]
})

# Salva os dados de previsão em arquivos CSV separados
previsoes_satelite.to_csv('previsoes_satelite.csv', index=False)
previsoes_sensores.to_csv('previsoes_sensores.csv', index=False)
previsoes_cultura.to_csv('previsoes_cultura.csv', index=False)
previsoes_estacao.to_csv('previsoes_estacao.csv', index=False)
previsoes_temporais.to_csv('previsoes_temporais.csv', index=False)

print("Previsões ficticias salvas em arquivos CSV separados.")
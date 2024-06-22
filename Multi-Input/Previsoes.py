import pandas as pd
import numpy as np

# Carrega as previsões de cada modelo (substitua pelos nomes dos seus arquivos)
previsoes_satelite = pd.read_csv('previsoes_satelite.csv')
previsoes_sensores = pd.read_csv('previsoes_sensores.csv')
previsoes_cultura = pd.read_csv('previsoes_cultura.csv')
previsoes_estacao = pd.read_csv('previsoes_estacao.csv')
previsoes_temporais = pd.read_csv('previsoes_temporais.csv')

# Combina as previsões em um único DataFrame
previsoes_combinadas = pd.concat([
    previsoes_satelite,
    previsoes_sensores,
    previsoes_cultura,
    previsoes_estacao,
    previsoes_temporais
], axis=1)

# Define os pesos para cada modelo em cada fase de desenvolvimento
pesos = {
    'Vegetativo': {
        'satelite': 0.1,
        'sensores': 0.3,
        'cultura': 0.5,
        'estacao': 0.1,
        'temporais': 0.0
    },
    'Floração': {
        'satelite': 0.1,
        'sensores': 0.2,
        'cultura': 0.6,
        'estacao': 0.1,
        'temporais': 0.0
    },
    'Colheita': {
        'satelite': 0.1,
        'sensores': 0.1,
        'cultura': 0.7,
        'estacao': 0.1,
        'temporais': 0.0
    }
}

# Cria uma coluna para o estágio de desenvolvimento
previsoes_combinadas['estagio_cultura'] = previsoes_combinadas['estagio_cultura'].astype('category')

# Calcula a previsão final utilizando a média ponderada
previsoes_combinadas['previsao_final'] = 0
for estagio in pesos:
    for modelo, peso in pesos[estagio].items():
        previsoes_combinadas.loc[previsoes_combinadas['estagio_cultura'] == estagio, 'previsao_final'] += previsoes_combinadas[modelo] * peso

# Salva as previsões finais
previsoes_combinadas.to_csv('previsoes_finais.csv', index=False)
print("Previsões finais salvas em previsões_finais.csv")
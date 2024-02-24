import matplotlib.pyplot as plt
import numpy as np

# Configurações iniciais
num_experimentos = 1000
ponto_inicial = 50  # Ajustando para iniciar no meio da escala
np.random.seed(0)  # Para reprodutibilidade

# Função para simular a eficácia dentro de 0 a 100
def simular_eficacia(ponto_inicial, variacao, num_experimentos):
    eficacia = np.cumsum(np.random.rand(num_experimentos) * variacao - variacao / 2) + ponto_inicial
    return np.clip(eficacia, 0, 100)  # Garantindo que os valores estejam entre 0 e 100

# Simulação da eficácia para representação conjunta e coordenada
variacao = 0.2
eficacia_conjunta_satelite = simular_eficacia(ponto_inicial, variacao, num_experimentos)
eficacia_conjunta_temporal = simular_eficacia(ponto_inicial, variacao, num_experimentos)
eficacia_conjunta_meteorologico = simular_eficacia(ponto_inicial, variacao, num_experimentos)
eficacia_conjunta_umidade = simular_eficacia(ponto_inicial, variacao, num_experimentos)

eficacia_coordenada_satelite = simular_eficacia(ponto_inicial, variacao, num_experimentos)
eficacia_coordenada_temporal = simular_eficacia(ponto_inicial, variacao, num_experimentos)
eficacia_coordenada_meteorologico = simular_eficacia(ponto_inicial, variacao, num_experimentos)
eficacia_coordenada_umidade = simular_eficacia(ponto_inicial, variacao, num_experimentos)

# Gráfico para Representação Conjunta
plt.figure(figsize=(14, 7))
plt.plot(eficacia_conjunta_satelite, label='Satélite', color='blue', linestyle='-')
plt.plot(eficacia_conjunta_temporal, label='Temporal', color='green', linestyle='-')
plt.plot(eficacia_conjunta_meteorologico, label='Meteorológico', color='red', linestyle='-')
plt.plot(eficacia_conjunta_umidade, label='Umidade', color='orange', linestyle='-')
plt.title('Eficácia da Representação Conjunta por Tipo de Entrada')
plt.xlabel('Número do Experimento')
plt.ylabel('Eficácia (%)')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

# Gráfico para Representação Coordenada
plt.figure(figsize=(14, 7))
plt.plot(eficacia_coordenada_satelite, label='Satélite', color='blue', linestyle='--')
plt.plot(eficacia_coordenada_temporal, label='Temporal', color='green', linestyle='--')
plt.plot(eficacia_coordenada_meteorologico, label='Meteorológico', color='red', linestyle='--')
plt.plot(eficacia_coordenada_umidade, label='Umidade', color='orange', linestyle='--')
plt.title('Eficácia da Representação Coordenada por Tipo de Entrada')
plt.xlabel('Número do Experimento')
plt.ylabel('Eficácia (%)')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.show()











import matplotlib.pyplot as plt
import numpy as np

# Simulando dados de exemplo para 4 tipos de entradas
# Supondo que cada tipo de entrada tenha um parâmetro chave que podemos medir
# Estes são valores simulados para fins de visualização

# Dados de satélite (por exemplo, brilho médio)
satellite_brightness = np.random.uniform(0.5, 1.0, 100)

# Dados temporais (por exemplo, variabilidade ao longo do tempo)
temporal_variability = np.random.uniform(0.1, 0.9, 100)

# Dados meteorológicos (por exemplo, índice de precipitação médio)
weather_precipitation = np.random.uniform(0.2, 1.0, 100)

# Dados de umidade do solo (por exemplo, nível médio de umidade)
soil_moisture = np.random.uniform(0.3, 0.8, 100)

# Criando o gráfico de dispersão
plt.figure(figsize=(10, 6))

# Scatter plots para cada tipo de entrada
plt.scatter(satellite_brightness, np.zeros_like(satellite_brightness) + 1, alpha=0.6, label='Satélite', color='blue')
plt.scatter(temporal_variability, np.zeros_like(temporal_variability) + 2, alpha=0.6, label='Temporal', color='red')
plt.scatter(weather_precipitation, np.zeros_like(weather_precipitation) + 3, alpha=0.6, label='Meteorológico', color='green')
plt.scatter(soil_moisture, np.zeros_like(soil_moisture) + 4, alpha=0.6, label='Umidade do Solo', color='orange')

# Ajustando os detalhes do gráfico
plt.yticks([1, 2, 3, 4], ['Satélite', 'Temporal', 'Meteorológico', 'Umidade do Solo'])
plt.ylabel('Tipo de Entrada')
plt.xlabel('Valor Simulado')
plt.title('')
plt.legend()

plt.show()

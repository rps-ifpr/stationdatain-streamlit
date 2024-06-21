import pandas as pd
import matplotlib.pyplot as plt

# Carrega os dados do arquivo CSV
dados = pd.read_csv('dados1975-2015.csv', sep=';', header=0, names=['Ano', 'Mes', 'Chuva', 'Evaporacao', 'Insolacao', 'TempMed', 'UmidRel', 'TempMaxAbs', 'TempMaxMed', 'TempMinAbs', 'TempMinMed'])

# Converte as colunas numéricas para float
dados[['Chuva', 'Evaporacao', 'Insolacao', 'TempMed', 'UmidRel', 'TempMaxAbs', 'TempMaxMed', 'TempMinAbs', 'TempMinMed']] = dados[['Chuva', 'Evaporacao', 'Insolacao', 'TempMed', 'UmidRel', 'TempMaxAbs', 'TempMaxMed', 'TempMinAbs', 'TempMinMed']].astype(float)

# 1. Análise Descritiva
print(dados.describe())

# 2. Identificação de Outliers
for coluna in dados.columns[2:]:
    plt.figure(figsize=(8, 4))
    plt.boxplot(dados[coluna], vert=False, patch_artist=True, showmeans=True)
    plt.title(f'Boxplot de {coluna}')
    plt.xlabel(coluna)
    plt.show()

# 3. Procurando por Padrões

# Cria um gráfico de dispersão para visualizar a relação entre variáveis
plt.figure(figsize=(8, 6))
plt.scatter(dados['TempMed'], dados['Chuva'], s=20, c='blue', alpha=0.5)
plt.xlabel('Temperatura Média')
plt.ylabel('Chuva')
plt.title('Relação entre Temperatura Média e Chuva')
plt.show()

# 4. Analisando outliers (opcional)

# Identifica valores que excedem um limite (por exemplo, 3 desvios padrões da média)
for coluna in dados.columns[2:]:
    media = dados[coluna].mean()
    desvio_padrao = dados[coluna].std()
    limite_superior = media + 3 * desvio_padrao
    limite_inferior = media - 3 * desvio_padrao
    outliers = dados[coluna][(dados[coluna] > limite_superior) | (dados[coluna] < limite_inferior)]
    print(f'Outliers em {coluna}: {outliers}')

# 5. Analisando padrões (opcional)

# Calcula a correlação entre as variáveis
correlacao = dados.corr()
print(correlacao)

# Cria um mapa de calor para visualizar a correlação
plt.figure(figsize=(10, 8))
plt.imshow(correlacao, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(dados.columns))
plt.xticks(tick_marks, dados.columns, rotation=45)
plt.yticks(tick_marks, dados.columns)
plt.title('Mapa de Calor de Correlação')
plt.show()
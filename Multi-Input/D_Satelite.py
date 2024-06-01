import ee
import os
import datetime
import requests
import geemap

# Credenciais da conta de serviço
service_account = 'Estes Dados Foram ocultados'
caminho_json = 'Estes Dados Foram ocultados'

# Autenticação
credentials = ee.ServiceAccountCredentials(service_account, caminho_json)
ee.Initialize(credentials)

# Função para baixar a imagem
def baixar_imagem(imagem, nome_arquivo):
    url = imagem.getDownloadURL({
        'name': nome_arquivo,
        'scale': 10,  # Ajuste a escala conforme necessário
        'region': geometry
    })
    print(f'Baixando: {nome_arquivo} - URL: {url}')
    response = requests.get(url, stream=True)
    with open(nome_arquivo, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# Função para pré-processar a imagem
def pre_processar_imagem(imagem):
    # 1. Correção Geométrica (opcional)
    # imagem = imagem.reproject(crs='EPSG:4326', scale=10)

    # 2. Máscara de Nuvens
    mascara_nuvens = imagem.select('CLOUDY_PIXEL_PERCENTAGE').lt(5)  # Ajuste o limite de nuvens
    imagem = imagem.updateMask(mascara_nuvens)

    # 3. Índices de Vegetação
    ndvi = imagem.normalizedDifference(['B8', 'B4']).rename('NDVI')
    imagem = imagem.addBands(ndvi)

    # 4. Cálculo de Bandas Adicionais (se necessário)
    # ...

    return imagem

# Coordenadas da área 22JBR (extraídas do MGRS_REF)
latitude_min = -27.107979524
latitude_max = -25.286086075548
longitude_min = -54.025311665
longitude_max = -53.9792316344072

# Criar um objeto Geometry com o quadrado
geometry = ee.Geometry.Rectangle([
    longitude_min,
    latitude_min,
    longitude_max,
    latitude_max
])

# Definir o intervalo de datas
data_inicio = '2023-09-01'
data_fim = '2023-12-01'

# Carregar as imagens (CORRIGIDO)
imagens = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(geometry) \
    .filterDate(ee.Date(data_inicio), ee.Date(data_fim)) \
    .sort('CLOUDY_PIXEL_PERCENTAGE') \
    .select('CLOUDY_PIXEL_PERCENTAGE')  # Selecione a banda antes do .first()


# Pré-processar as imagens
imagens_processadas = imagens.map(pre_processar_imagem)

# Baixar as imagens pré-processadas
for i in range(imagens_processadas.size().getInfo()):
    imagem = ee.Image(imagens_processadas.toList(imagens_processadas.size()).get(i))
    data = ee.Date(imagem.get('system:time_start')).format('YYYY-MM-dd').getInfo()
    nome_arquivo = f'imagem_{data}.tif'
    baixar_imagem(imagem, nome_arquivo)

# Exibir mensagem de conclusão
print(f'Download de {imagens_processadas.size().getInfo()} imagens concluído!')

# Visualização com geemap
Map = geemap.Map()  # Cria um mapa interativo
Map.addLayer(imagens_processadas, {}, 'Imagem Processada')
Map.centerObject(geometry, 10) # Centra o mapa na área
Map.show()  # Exibe o mapa

# Este código realiza a baixar pre-processa as imagens da area de trabalho
import streamlit as st
import ee
import os
import datetime
import requests
from geopandas import clip


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

# Interface Streamlit
st.title('Aplicação Streamlit com Earth Engine')

# Coordenadas do ponto central
latitude = -25.68336105699554
longitude = -53.786481243561795

# Distância a partir do ponto central para formar o quadrado (em metros)
distancia = 800

# Criar um objeto Geometry com o quadrado
geometry = ee.Geometry.Rectangle([
    longitude - distancia / 111320,
    latitude - distancia / 111320,
    longitude + distancia / 111320,
    latitude + distancia / 111320
])

# Definir o intervalo de datas
data_inicio = '2023-09-01'
data_fim = '2023-12-01'

# Carregar as imagens
imagens = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(geometry) \
    .filterDate(ee.Date(data_inicio), ee.Date(data_fim)) \
    .sort('CLOUDY_PIXEL_PERCENTAGE')

# Pegar a primeira imagem da coleção
imagem = ee.Image(imagens.first())

# Pré-processar a imagem
imagem_processada = pre_processar_imagem(imagem)

# Baixar a imagem pré-processada
data = ee.Date(imagem.get('system:time_start')).format('YYYY-MM-dd').getInfo()
nome_arquivo = f'imagem_{data}.tif'
baixar_imagem(imagem_processada, nome_arquivo)

# Exibir mensagem de conclusão
st.success(f'Download da imagem concluído!')
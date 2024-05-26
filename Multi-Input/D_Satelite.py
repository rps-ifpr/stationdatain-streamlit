import streamlit as st
import ee
import json


# Autenticação
with open(caminho_json) as json_file:
    credentials = ee.ServiceAccountCredentials(service_account, json_file.read())

# Inicialização da API do Earth Engine
ee.Initialize(credentials)  # Inicializa a API do Earth Engine com as credenciais

# Função para exibir a imagem com resolução ajustada e realce
def exibir_imagem(imagem):
    # Realce de nitidez (opcional - ajuste os parâmetros conforme necessário)
    sharpened_image = imagem.convolve(ee.Kernel.gaussian(radius=1, sigma=1, magnitude=3)).subtract(
        imagem.convolve(ee.Kernel.gaussian(radius=1, sigma=1))
    )

    vis_params = {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 3000,
        'gamma': 1.8
    }
    url = sharpened_image.getThumbUrl({
        'dimensions': 1024,
        'format': 'png',
        **vis_params
    })
    st.image(url)

# Interface Streamlit
st.title('Aplicação Streamlit com Earth Engine')

# Coordenadas do ponto central
latitude = -25.68336105699554
longitude = -53.786481243561795

# Distância a partir do ponto central para formar o quadrado (em metros)
distancia = 600  # Ajuste este valor para controlar o tamanho do quadrado

# Criar um objeto Geometry com o quadrado
geometry = ee.Geometry.Rectangle([
    longitude - distancia / 111320,
    latitude - distancia / 111320,
    longitude + distancia / 111320,
    latitude + distancia / 111320
])

# Carregar a imagem Sentinel-2 mais recente do dia
imagem = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(geometry) \
    .sort('CLOUDY_PIXEL_PERCENTAGE') \
    .first() \
    .clip(geometry)

# Exibir a imagem com resolução melhorada
exibir_imagem(imagem)

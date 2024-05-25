<<<<<<< HEAD
# Este código encontra as imagens de posição de area de trabalho
import streamlit as st
=======
>>>>>>> parent of 81f839d (UpDate)
import ee

<<<<<<< HEAD
# Credenciais da conta de serviço

=======
# Seu ID de cliente (substitua pelo seu ID real)
client_id = '785957889466-i3l8rrgf64lb9lrdigbir3ev4jp6vt1j.apps.googleusercontent.com'
>>>>>>> parent of 81f839d (UpDate)

# Autenticação com o ID do cliente
ee.Authenticate(client_id=client_id)

<<<<<<< HEAD
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
        'gamma': 1.0
    }
    url = sharpened_image.getThumbUrl({
        'dimensions': 1800,
        'format': 'png',
        **vis_params
    })
    st.image(url)
=======
# Inicialização
ee.Initialize()
>>>>>>> parent of 81f839d (UpDate)

# Teste de acesso à API
try:
    # Tenta carregar uma imagem de exemplo
    imagem = ee.Image('USGS/SRTMGL1_003')
    print('Acesso à API confirmado! Informações da imagem:')
    print(imagem.getInfo())

<<<<<<< HEAD
# Coordenadas do ponto central
latitude = -25.68336105699554
longitude = -53.786481243561795

# Distância a partir do ponto central para formar o quadrado (em metros)
distancia = 800  # Ajuste este valor para controlar o tamanho do quadrado

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
=======
except ee.EEException as e:
    print(f'Erro ao acessar a API: {e}')
>>>>>>> parent of 81f839d (UpDate)

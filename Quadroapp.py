import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio

def load_real_data(filepath):
    with rasterio.open(filepath) as src:
        # Lê a imagem. O índice 'read' começa em 1. Aqui, estamos lendo todas as bandas.
        image = src.read()
        # Se a sua imagem tiver bandas específicas a serem visualizadas ou precisar de tratamento especial,
        # você pode ajustar a leitura aqui.

        # Transpondo de (bandas, altura, largura) para (altura, largura, bandas) para visualização
        image = np.transpose(image, (1, 2, 0))
        # Se sua imagem for monocromática ou precisar de uma combinação específica de bandas,
        # ajuste a transposição ou manipulação de bandas aqui.

    return image

# Atualize o caminho para a localização da sua imagem específica, se diferente.
filepath = "Multi-Input/Sentinel2_Image7.tif"

# Carregar dados reais
X_satellite_real = load_real_data(filepath)

# Visualizando uma amostra dos Dados de Satélite
st.write("Visualizando uma imagem de satélite real:")
fig, ax = plt.subplots()
# Se sua imagem tiver bandas que não são RGB padrão ou precisar de processamento de cores,
# ajuste a visualização aqui.
ax.imshow(X_satellite_real[:, :, :3])  # Isso pressupõe que as primeiras 3 bandas possam ser visualizadas como RGB.
st.pyplot(fig)

import streamlit as st
import pandas as pd

# Exemplo de dados de localização
data = {
    'lat': [37.7749, 34.0522, 40.7128],
    'lon': [-122.4194, -118.2437, -74.0060],
    'nome': ['São Francisco', 'Los Angeles', 'Nova York']
}
df = pd.DataFrame(data)

# Visualizando os dados no mapa
st.map(df)

import streamlit as st
import pydeck as pdk

# Exemplo de visualização de dados no mapa com pydeck
# Define a camada de dados
layer = pdk.Layer(
    'ScatterplotLayer',     # Tipo da camada
    df,                     # DataFrame com os dados
    get_position=['lon', 'lat'],  # Colunas do DataFrame que contêm as coordenadas
    get_color='[200, 30, 0, 160]',  # Cor dos pontos
    get_radius=10000,          # Tamanho dos pontos
)

# Define a visualização
view_state = pdk.ViewState(
    latitude=37.7749,
    longitude=-122.4194,
    zoom=4,
    pitch=0
)

# Cria e mostra a visualização
r = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9')
st.pydeck_chart(r)

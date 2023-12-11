import ee
import geemap
import folium

# Inicialize a API do Google Earth Engine
ee.Initialize()

# Defina a região de interesse
geometry = ee.Geometry.Polygon(
    [[[-53.78835325312135, -25.68427745715154],
      [-53.78531699251649, -25.68517665067112],
      [-53.784973669762586, -25.685979150983904],
      [-53.78345017504213, -25.685002613745393],
      [-53.78253822397706, -25.683387925701712],
      [-53.78385250639436, -25.683199382856568],
      [-53.78692588072439, -25.68231893708208],
      [-53.787065355593164, -25.68272503199412],
      [-53.78782710295339, -25.682521984711116],
      [-53.78835325312135, -25.68427745715154]]])

# Defina a coleção de dados climáticos (ERA5-Land)
dataset = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
    .select(['temperature_2m', 'total_precipitation']) \
    .filterBounds(geometry) \
    .filterDate('2022-01-01', '2023-11-01')

# Reduza a coleção para obter médias diárias
dailyData = dataset.mean()

# Crie um mapa interativo usando geemap
Map = geemap.Map()
Map.centerObject(geometry, 8)

# Adicione a imagem ao mapa
Map.addLayer(dailyData, {
    'bands': ['temperature_2m', 'total_precipitation'],
    'min': [-30, 0],
    'max': [30, 20],
    'opacity': 0.7
}, 'Dados Climáticos (ERA5-Land)')

# Adicione a região de interesse ao mapa
Map.addLayer(geometry, {}, 'Região de Interesse')

# Exiba o mapa
Map

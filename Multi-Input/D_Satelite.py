import ee
import geemap
import streamlit as st
import rasterio
import geopandas as gpd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ee.Initialize()

# Define a área de interesse como um polígono
aoi = ee.Geometry.Polygon([[
    [-50.0, -20.0],
    [-50.0, -10.0],
    [-40.0, -10.0],
    [-40.0, -20.0],
    [-50.0, -20.0]
]])

# Carrega uma imagem Sentinel-2
image = ee.ImageCollection('COPERNICUS/S2').filterDate('2023-01-01', '2023-01-31').filterBounds(aoi).first()


st.title('Visualização da Imagem de Satélite')
st.map(geemap.ee_to_geojson(image.geometry()))
st.image(geemap.ee_to_base64(image.visualize(**{'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000})))
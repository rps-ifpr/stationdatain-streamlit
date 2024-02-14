import streamlit as st
import folium
from folium import Marker

from pyowm import OWM

def get_weather(city):
    # Substitua 'sua_chave_de_api' pela sua chave de API da OpenWeatherMap
    owm = OWM('342e60213afbec5b9b52e7f87d90248e')
    mgr = owm.weather_manager()
    observation = mgr.weather_at_place(city)
    weather = observation.weather

    return {
        'condition': weather.detailed_status,
        'temperature': weather.temperature('celsius')['temp'],
        'humidity': weather.humidity
    }

st.title('Mapa do Local com Informações Meteorológicas')

city = st.text_input('Digite o nome da cidade e o código do país (por exemplo, "Capanema, BR"):')
if city:
    weather_data = get_weather(city)

    # Cria um mapa centrado na localização da cidade
    m = folium.Map(location=[observation.location.lat, observation.location.lon], zoom_start=10)

    # Adiciona um marcador para a localização da cidade
    Marker([observation.location.lat, observation.location.lon], popup=city).add_to(m)

    # Exibe as informações meteorológicas no popup do marcador
    popup_text = f"Condição: {weather_data['condition']}<br>Temperatura (°C): {weather_data['temperature']}<br>Umidade (%): {weather_data['humidity']}"
    folium.Popup(popup_text).add_to(m)

    # Exibe o mapa
    folium_static(m)

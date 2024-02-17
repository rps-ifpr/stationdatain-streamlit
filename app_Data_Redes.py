import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, LSTM, Concatenate, Flatten, BatchNormalization, Dropout, MaxPooling2D

# Configuração do Streamlit
st.title ('Treinamento de Modelo e Visualização de Métricas')

# Gerar Dados Sintéticos de Entrada
def generate_synthetic_data(num_samples=10):
    X_satellite = np.random.rand(num_samples, 128, 128, 3)  # Imagens de satélite
    X_temporal = np.random.rand(num_samples, 30, 1)  # Dados de séries temporais
    X_weather = np.random.rand(num_samples, 10)  # Dados meteorológicos
    X_soil = np.random.rand(num_samples, 5)  # Dados de umidade do solo
    X_crop_info = np.random.rand(num_samples, 7)  # Informações sobre as culturas
    Y = np.random.randint(2, size=(num_samples, 1))  # Saídas binárias (0 ou 1)
    return X_satellite, X_temporal, X_weather, X_soil, X_crop_info, Y

# Definição das Sub-Redes e Modelo Completo
def satellite_subnetwork(input_shape):
    input_layer = Input (shape=input_shape)
    x = Conv2D (64, (3, 3), padding='same', activation='relu') (input_layer)
    x = BatchNormalization () (x)
    x = MaxPooling2D (pool_size=(2, 2)) (x)
    x = Conv2D (128, (3, 3), padding='same', activation='relu') (x)
    x = BatchNormalization () (x)
    x = MaxPooling2D (pool_size=(2, 2)) (x)
    x = Flatten () (x)
    x = Dropout (0.5) (x)
    return Model (inputs=input_layer, outputs=x, name='satellite_subnetwork')


def temporal_subnetwork(input_shape):
    input_layer = Input (shape=input_shape)
    x = LSTM (50, return_sequences=True) (input_layer)
    x = LSTM (50) (x)
    x = Dropout (0.5) (x)
    return Model (inputs=input_layer, outputs=x, name='temporal_subnetwork')


def weather_subnetwork(input_shape):
    input_layer = Input (shape=input_shape)
    x = Dense (64, activation='relu') (input_layer)
    x = BatchNormalization () (x)
    x = Dropout (0.5) (x)
    x = Dense (64, activation='relu') (x)
    x = BatchNormalization () (x)
    x = Dropout (0.5) (x)
    return Model (inputs=input_layer, outputs=x, name='weather_subnetwork')


def soil_moisture_subnetwork(input_shape):
    input_layer = Input (shape=input_shape)
    x = Dense (64, activation='relu') (input_layer)
    x = BatchNormalization () (x)
    x = Dropout (0.5) (x)
    x = Dense (64, activation='relu') (x)
    x = BatchNormalization () (x)
    x = Dropout (0.5) (x)
    return Model (inputs=input_layer, outputs=x, name='soil_moisture_subnetwork')


def crop_info_subnetwork(input_shape):
    input_layer = Input (shape=input_shape)
    x = Dense (64, activation='relu') (input_layer)
    x = BatchNormalization () (x)
    x = Dropout (0.5) (x)
    x = Dense (64, activation='relu') (x)
    x = BatchNormalization () (x)
    x = Dropout (0.5) (x)
    return Model (inputs=input_layer, outputs=x, name='crop_info_subnetwork')


def create_complete_model(sat_shape, temp_shape, weather_shape, soil_shape, crop_info_shape):
    satellite_input = Input (shape=sat_shape, name='satellite_input')
    temporal_input = Input (shape=temp_shape, name='temporal_input')
    weather_input = Input (shape=weather_shape, name='weather_input')
    soil_input = Input (shape=soil_shape, name='soil_moisture_input')
    crop_info_input = Input (shape=crop_info_shape, name='crop_info_input')

    satellite_net = satellite_subnetwork (sat_shape) (satellite_input)
    temporal_net = temporal_subnetwork (temp_shape) (temporal_input)
    weather_net = weather_subnetwork (weather_shape) (weather_input)
    soil_net = soil_moisture_subnetwork (soil_shape) (soil_input)
    crop_info_net = crop_info_subnetwork (crop_info_shape) (crop_info_input)

    merged = Concatenate () ([satellite_net, temporal_net, weather_net, soil_net, crop_info_net])
    decision_layer = Dense (1, activation='sigmoid', name='decision_layer') (merged)

    model = Model (inputs=[satellite_input, temporal_input, weather_input, soil_input, crop_info_input],
                   outputs=decision_layer)
    return model


# Compilação e Treinamento do Modelo
@st.cache_data  # Usando o novo decorador recomendado st.cache_data
def train_model():
    X_satellite, X_temporal, X_weather, X_soil, X_crop_info, Y = generate_synthetic_data (10)
    model = create_complete_model ((128, 128, 3), (30, 1), (10,), (5,), (7,))
    model.compile (optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit ([X_satellite, X_temporal, X_weather, X_soil, X_crop_info], Y, epochs=10, batch_size=32,
                         validation_split=0.2)
    return history


if st.button ('Treinar Modelo'):
    with st.spinner ('Treinando...'):
        history = train_model ()

        # Gerando gráficos para perda e acurácia
        fig, ax = plt.subplots (2, 1, figsize=(10, 10))
        ax[0].plot (history.history['loss'], label='Perda de Treinamento')
        ax[0].plot (history.history['val_loss'], label='Perda de Validação')
        ax[0].set_xlabel ('Épocas')
        ax[0].set_ylabel ('Perda')
        ax[0].set_title ('Perda por Época')
        ax[0].legend ()

        ax[1].plot (history.history['accuracy'], label='Acurácia de Treinamento')
        ax[1].plot (history.history['val_accuracy'], label='Acurácia de Validação')
        ax[1].set_xlabel ('Épocas')
        ax[1].set_ylabel ('Acurácia')
        ax[1].set_title ('Acurácia por Época')
        ax[1].legend ()

        st.pyplot (fig)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, LSTM, Concatenate, Flatten, BatchNormalization, Dropout, MaxPooling2D
from data_generation import generate_synthetic_data

def satellite_subnetwork(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    return Model(inputs=input_layer, outputs=x, name='satellite_subnetwork')

# As definições para temporal_subnetwork, weather_subnetwork, soil_moisture_subnetwork, e crop_info_subnetwork permanecem as mesmas.

def create_complete_model():
    sat_shape, temp_shape, weather_shape, soil_shape, crop_info_shape = (128, 128, 3), (4320, 1), (10,), (5,), (7,)
    satellite_input = Input(shape=sat_shape, name='satellite_input')
    temporal_input = Input(shape=temp_shape, name='temporal_input')
    weather_input = Input(shape=weather_shape, name='weather_input')
    soil_input = Input(shape=soil_shape, name='soil_moisture_input')
    crop_info_input = Input(shape=crop_info_shape, name='crop_info_input')

    satellite_net = satellite_subnetwork(sat_shape)(satellite_input)
    temporal_net = temporal_subnetwork(temp_shape)(temporal_input)
    weather_net = weather_subnetwork(weather_shape)(weather_input)
    soil_net = soil_moisture_subnetwork(soil_shape)(soil_input)
    crop_info_net = crop_info_subnetwork(crop_info_shape)(crop_info_input)

    merged = Concatenate()([satellite_net, temporal_net, weather_net, soil_net, crop_info_net])
    decision_layer = Dense(1, activation='sigmoid', name='decision_layer')(merged)

    model = Model(inputs=[satellite_input, temporal_input, weather_input, soil_input, crop_info_input], outputs=decision_layer)
    return model

def train_model(num_samples=10):
    X_satellite, X_temporal, X_weather, X_soil, X_crop_info, Y = generate_synthetic_data(num_samples)
    model = create_complete_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit([X_satellite, X_temporal, X_weather, X_soil, X_crop_info], Y, epochs=10, batch_size=32, validation_split=0.2)
    return model, history

import streamlit as st
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, LSTM, Concatenate, Flatten, BatchNormalization, Dropout, MaxPooling2D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model

# Configuração do Streamlit
st.title('Treinamento de Modelo e Visualização de Métricas')

# Definição da função generate_synthetic_data
def generate_synthetic_data(num_samples=10):
    num_hours = 30 * 24 * 12  # 4320 pontos de dados para simular entradas a cada hora durante 6 meses
    X_satellite = np.random.rand(num_samples, 128, 128, 3)  # Imagens de satélite
    X_temporal = np.random.rand(num_samples, 4320, 1)  # Dados de séries temporais ajustados
    X_weather = np.random.rand(num_samples, 10)  # Dados meteorológicos
    X_soil = np.random.rand(num_samples, 5)  # Dados de umidade do solo
    X_crop_info = np.random.rand(num_samples, 7)  # Informações sobre as culturas
    Y = np.random.randint(2, size=(num_samples, 1))  # Saídas binárias (0 ou 1)
    return X_satellite, X_temporal, X_weather, X_soil, X_crop_info, Y


# Botão para gerar e visualizar dados sintéticos
if st.button ('Gerar e Visualizar Dados Sintéticos'):
    num_samples = 10000  # Define o número de amostras que você quer gerar
    X_satellite, X_temporal, X_weather, X_soil, X_crop_info, Y = generate_synthetic_data (num_samples)

    # Visualizando uma amostra dos Dados de Satélite
    st.write ("Visualizando uma imagem de satélite sintética:")
    idx = 0  # Index da amostra para visualização
    fig, ax = plt.subplots ()
    ax.imshow (X_satellite[idx])
    st.pyplot (fig)

    # Visualizando Dados Temporais (primeira série temporal)
    st.write ("Visualizando dados temporais sintéticos (primeira série temporal):")
    st.line_chart (X_temporal[idx].flatten ())

    # Visualizando Dados Meteorológicos como DataFrame
    st.write ("Dados Meteorológicos Sintéticos:")
    df_weather = pd.DataFrame (X_weather, columns=[f'Feature_{i + 1}' for i in range (X_weather.shape[1])])
    st.dataframe (df_weather)

    # Visualizando Dados de Umidade do Solo como DataFrame
    st.write ("Dados de Umidade do Solo Sintéticos:")
    df_soil = pd.DataFrame (X_soil, columns=[f'Feature_{i + 1}' for i in range (X_soil.shape[1])])
    st.dataframe (df_soil)

    # Visualizando Informações sobre Culturas como DataFrame
    st.write ("Informações sobre Culturas Sintéticas:")
    df_crop_info = pd.DataFrame (X_crop_info, columns=[f'Feature_{i + 1}' for i in range (X_crop_info.shape[1])])
    st.dataframe (df_crop_info)

    # Visualizando as Saídas Binárias (Labels) como DataFrame
    st.write ("Labels (Saídas Binárias) Sintéticas:")
    df_labels = pd.DataFrame (Y, columns=['Label'])
    st.dataframe (df_labels)

# Definição das Sub-Redes
def satellite_subnetwork(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='sat_conv1')(input_layer)  # Nome único para a primeira camada Conv2D
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='sat_conv2')(x)  # Nome único para a segunda camada Conv2D
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    return Model(inputs=input_layer, outputs=x, name='satellite_subnetwork')

def temporal_subnetwork(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(50, return_sequences=True, name='temp_lstm1')(input_layer)  # Nome único para a primeira camada LSTM
    x = LSTM(50, name='temp_lstm2')(x)  # Nome único para a segunda camada LSTM
    x = Dropout(0.5)(x)
    return Model(inputs=input_layer, outputs=x, name='temporal_subnetwork')

def weather_subnetwork(input_shape):
    input_layer = Input(shape=input_shape)
    x = Dense(64, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return Model(inputs=input_layer, outputs=x, name='weather_subnetwork')

def soil_moisture_subnetwork(input_shape):
    input_layer = Input(shape=input_shape)
    x = Dense(64, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return Model(inputs=input_layer, outputs=x, name='soil_moisture_subnetwork')

def crop_info_subnetwork(input_shape):
    input_layer = Input(shape=input_shape)
    x = Dense(64, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return Model(inputs=input_layer, outputs=x, name='crop_info_subnetwork')

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

# Função para treinar o modelo e retornar modelo e histórico
@st.cache_data
def train_model():
    X_satellite, X_temporal, X_weather, X_soil, X_crop_info, Y = generate_synthetic_data(10)
    model = create_complete_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit([X_satellite, X_temporal, X_weather, X_soil, X_crop_info], Y, epochs=10, batch_size=32, validation_split=0.2)
    return model, history

# Função para gerar dados de validação, idêntica à geração de dados sintéticos
def generate_validation_data(num_samples=2):
    return generate_synthetic_data(num_samples)

# Função para plotar métricas adicionais após o treinamento
def plot_additional_metrics(model, X_test, Y_test):
    pred_probs = model.predict(X_test)
    fpr, tpr, _ = roc_curve(Y_test.ravel(), pred_probs.ravel())
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Receiver Operating Characteristic')
    ax[0].legend(loc="lower right")

    pred_classes = (pred_probs > 0.5).astype(int)
    cm = confusion_matrix(Y_test, pred_classes)
    im = ax[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax[1].figure.colorbar(im, ax=ax[1])
    ax[1].set(xticks=np.arange(cm.shape[1]),
              yticks=np.arange(cm.shape[0]),
              title='Confusion Matrix',
              ylabel='True label',
              xlabel='Predicted label')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[1].text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    st.pyplot(fig)

# Variáveis globais para armazenar o modelo e o histórico após o treinamento
model = None
history = None

# Use st.session_state para armazenar e acessar o modelo e o histórico
if 'model' not in st.session_state or 'history' not in st.session_state:
    st.session_state.model = None
    st.session_state.history = None

# Botão para iniciar o treinamento do modelo
if st.button('Treinar Modelo'):
    with st.spinner('Treinando...'):
        st.session_state.model, st.session_state.history = train_model()
        st.success('Treinamento concluído!')

if st.button ('Visualizar Métricas de Treinamento', key='view_training_metrics'):
    if st.session_state.history:
        fig, ax = plt.subplots (2, 1, figsize=(12, 10))

        epochs = range (1, len (st.session_state.history.history['loss']) + 1)
        # Supondo uma relação direta de 1 época por mês para simplificar
        months = epochs  # Ajuste conforme necessário para o seu caso

        ax[0].plot (epochs, st.session_state.history.history['loss'], label='Perda de Treinamento')
        ax[0].plot (epochs, st.session_state.history.history['val_loss'], label='Perda de Validação')
        ax[0].set_xlabel ('Meses')  # Alterado de 'Épocas' para 'Meses'
        ax[0].set_ylabel ('Perda')
        ax[0].set_title ('Perda por Mês')  # Alterado para refletir 'por Mês'
        ax[0].legend ()

        ax[1].plot (epochs, st.session_state.history.history['accuracy'], label='Acurácia de Treinamento')
        ax[1].plot (epochs, st.session_state.history.history['val_accuracy'], label='Acurácia de Validação')
        ax[1].set_xlabel ('Meses')  # Alterado de 'Épocas' para 'Meses'
        ax[1].set_ylabel ('Acurácia')
        ax[1].set_title ('Acurácia por Mês')  # Alterado para refletir 'por Mês'
        ax[1].legend ()

        # Adicionando linhas verticais para indicar a passagem de meses
        for month in months:
            ax[0].axvline (x=month, color='gray', linestyle='--', alpha=0.5)
            ax[1].axvline (x=month, color='gray', linestyle='--', alpha=0.5)

        st.pyplot (fig)
    else:
        st.write ("Por favor, treine o modelo primeiro para visualizar as métricas.")

# Supondo que X_real e X_simulated sejam seus dados reais e simulados
# feature_index é o índice da característica que você deseja analisar

def compare_distributions(X_real, X_simulated, feature_index=0):
    mean_real = np.mean(X_real[:, feature_index])
    std_real = np.std(X_real[:, feature_index])
    mean_simulated = np.mean(X_simulated[:, feature_index])
    std_simulated = np.std(X_simulated[:, feature_index])

    st.write(f"Real Data: Mean = {mean_real}, Std = {std_real}")
    st.write(f"Simulated Data: Mean = {mean_simulated}, Std = {std_simulated}")

    # Plot
    fig, ax = plt.subplots()
    ax.hist(X_real[:, feature_index], alpha=0.5, label='Real')
    ax.hist(X_simulated[:, feature_index], alpha=0.5, label='Simulated')
    ax.legend(loc='upper right')
    ax.set_title("Comparison of Data Distributions")
    st.pyplot(fig)

# Supondo que create_complete_model() crie seu modelo TensorFlow/Keras

def evaluate_model_performance(X_real, Y_real, X_simulated, Y_simulated, create_complete_model):
    model = create_complete_model()

    # Treinamento com dados simulados
    X_train_sim, X_test_sim, Y_train_sim, Y_test_sim = train_test_split(X_simulated, Y_simulated, test_size=0.2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_sim, Y_train_sim, epochs=10, batch_size=32, validation_split=0.2)

    # Avaliação em dados simulados
    Y_pred_sim = model.predict(X_test_sim) > 0.5
    accuracy_sim = accuracy_score(Y_test_sim, Y_pred_sim)
    st.write(f"Accuracy on Simulated Test Data: {accuracy_sim}")

    # Avaliação em dados reais
    X_train_real, X_test_real, Y_train_real, Y_test_real = train_test_split(X_real, Y_real, test_size=0.2)
    Y_pred_real = model.predict(X_test_real) > 0.5
    accuracy_real = accuracy_score(Y_test_real, Y_pred_real)
    st.write(f"Accuracy on Real Test Data: {accuracy_real}")

# Definição dos dados reais e simulados (substitua com seus próprios dados)
X_real, Y_real = np.random.rand(100, 5), np.random.randint(2, size=100)
X_simulated, Y_simulated = np.random.rand(100, 5), np.random.randint(2, size=100)




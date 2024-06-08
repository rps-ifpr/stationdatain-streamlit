import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random


# 1. Carregar as imagens pré-processadas e labels:
def carregar_dados(diretorio_imagens, diretorio_labels):
    """
    Carrega as imagens pré-processadas e as labels.
    """
    image_paths = [
        os.path.join (diretorio_imagens, f)
        for f in os.listdir (diretorio_imagens)
        if f.lower ().endswith (('.jpg', '.png', '.tiff'))
    ]
    labels = []
    for i, filename in enumerate (image_paths):
        label_file = os.path.join (diretorio_labels, os.path.splitext (filename)[0] + ".txt")
        if os.path.exists (label_file):
            with open (label_file, "r") as f:
                label = int (f.readline ().strip ())
                labels.append (label)
        else:
            print (f"Arquivo de label não encontrado para {filename}")

    X = np.array ([cv2.imread (image_path) for image_path in image_paths])
    y = np.array (labels)
    return X, y


# 2. Pré-processar os dados (se necessário):
def preprocess_data(X):
    """
    Pré-processa os dados de imagens.
    """
    X = X.astype ('float32') / 255.0  # Normalização
    return X


# 3. Criar o modelo CNN:
def criar_modelo(input_shape, num_classes):
    """
    Cria um modelo de CNN simples.
    """
    model = Sequential ()
    model.add (Conv2D (32, (3, 3), activation='relu', input_shape=input_shape))
    model.add (MaxPooling2D ((2, 2)))
    model.add (Conv2D (64, (3, 3), activation='relu'))
    model.add (MaxPooling2D ((2, 2)))
    model.add (Flatten ())
    model.add (Dense (128, activation='relu'))
    model.add (Dense (num_classes, activation='softmax'))
    return model


# 4. Treinar o modelo:
def treinar_modelo(X, y, num_classes, epochs=10, batch_size=32):
    """
    Treina o modelo CNN.
    """
    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)
    y_train = to_categorical (y_train, num_classes=num_classes)
    y_test = to_categorical (y_test, num_classes=num_classes)

    model = criar_modelo (X_train.shape[1:], num_classes)
    model.compile (optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit (X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    return model, history


# 5. Avaliar o modelo:
def avaliar_modelo(model, X_test, y_test, num_classes):
    """
    Avalia o modelo CNN.
    """
    y_test = to_categorical (y_test, num_classes=num_classes)
    loss, accuracy = model.evaluate (X_test, y_test)
    print (f"Perda: {loss:.4f}")
    print (f"Acurácia: {accuracy:.4f}")


# 6. Visualizar os resultados:
def visualizar_resultados(history):
    """
    Visualiza a acurácia e perda durante o treinamento.
    """
    plt.figure (figsize=(12, 6))
    plt.subplot (1, 2, 1)
    plt.plot (history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot (history.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel ('Épocas')
    plt.ylabel ('Acurácia')
    plt.legend ()

    plt.subplot (1, 2, 2)
    plt.plot (history.history['loss'], label='Perda de Treinamento')
    plt.plot (history.history['val_loss'], label='Perda de Validação')
    plt.xlabel ('Épocas')
    plt.ylabel ('Perda')
    plt.legend ()

    plt.tight_layout ()
    plt.show ()


# Carrega os dados pré-processados e as labels
X, y = carregar_dados ("C:/IFPR-CONTEUDO/GITHUB/stationdatain-streamlit/Multi-Input/preprocessed_images",
                       "C:/IFPR-CONTEUDO/GITHUB/stationdatain-streamlit/Multi-Input/preprocessed_images/labels")

# Pré-processa os dados 
X = preprocess_data (X)

# Treina o modelo
model, history = treinar_modelo (X, y, num_classes=2, epochs=10, batch_size=32)

# Avalia o modelo
avaliar_modelo (model, X_test, y_test, num_classes=2)

# Visualiza os resultados
visualizar_resultados (history)
import os
import cv2
import numpy as np
from PIL import Image
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Diretório de entrada com as imagens
input_dir = "C:/IFPR-CONTEUDO/GITHUB/stationdatain-streamlit/Multi-Input/preprocessed_images"

# Diretório para os arquivos de label
labels_dir = "C:/IFPR-CONTEUDO/GITHUB/stationdatain-streamlit/Multi-Input/preprocessed_images/labels"

# Tamanho do conjunto de dados
num_images = 21

# Carregar as imagens e labels
def carregar_dados(input_dir, labels_dir, num_images):
    """
    Carrega as imagens e labels do conjunto de dados.
    """
    X = []
    y = []
    for i in range(num_images):
        image_path = os.path.join(input_dir, f"jpg_{i}_preprocessed.jpg")
        label_path = os.path.join(labels_dir, f"jpg_{i}_preprocessed.txt")

        # Carregar a imagem
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        X.append(image)

        # Carregar o label
        with open(label_path, 'r') as f:
            label = int(f.readline())
        y.append(label)

    return np.array(X), np.array(y)

# Criar o modelo CNN
def criar_modelo_cnn():
    """
    Cria o modelo de rede neural convolucional (CNN).
    """
    model = Sequential()

    # Camada convolucional 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))

    # Camada convolucional 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Camada convolucional 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Camada de Flatten
    model.add(Flatten())

    # Camada densa 1
    model.add(Dense(128, activation='relu'))

    # Camada de saída (com dois neurônios para duas classes)
    model.add(Dense(2, activation='softmax'))

    # Compilar o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Carregar os dados
X, y = carregar_dados(input_dir, labels_dir, num_images)

# Converter labels para formato one-hot encoding
y = to_categorical(y, num_classes=2)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo CNN
modelo_cnn = criar_modelo_cnn()  # Cria o modelo

# Definir o EarlyStopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinar o modelo
modelo_cnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Avaliar o modelo
_, accuracy = modelo_cnn.evaluate(X_test, y_test)  # Avalia o modelo
print(f"Acurácia do modelo: {accuracy:.4f}")

# Predições no conjunto de teste
y_pred = modelo_cnn.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calcular a matriz de confusão
confusion_mtx = confusion_matrix(y_true_classes, y_pred_classes)

# Plota a matriz de confusão
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.colorbar()
tick_marks = np.arange(len(set(y_true_classes)))
plt.xticks(tick_marks, set(y_true_classes))
plt.yticks(tick_marks, set(y_true_classes))
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.show()

# Salvar o modelo
modelo_cnn.save("modelo_cnn_irrigacao.h5")
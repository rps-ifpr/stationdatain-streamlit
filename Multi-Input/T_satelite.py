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
<<<<<<< HEAD
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
=======
from sklearn.metrics import confusion_matrix
>>>>>>> parent of 444c522 (Create modelo_cnn_irrigacao.h5)
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
history = modelo_cnn.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1) # Adiciona verbose=1

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

<<<<<<< HEAD

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'r--')  # Linha de referência
fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
plt.plot(fpr, tpr, label='Curva ROC')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend()
plt.show()

# Mostrar gráfico da perda e acurácia
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda de Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia de Treinamento e Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.tight_layout()
plt.show()

# Calcular precisão, revocação e F1-score
precision = precision_score(y_true_classes, y_pred_classes)
recall = recall_score(y_true_classes, y_pred_classes)
f1 = f1_score(y_true_classes, y_pred_classes)

# Mostrar os resultados
st.title("Avaliação do Modelo CNN")
st.write(f"Acurácia: {accuracy:.4f}")
st.write(f"Precisão: {precision:.4f}")
st.write(f"Revocação: {recall:.4f}")
st.write(f"F1-Score: {f1:.4f}")

# Criar um gráfico de barras para as métricas
metrics = ['Acurácia', 'Precisão', 'Revocação', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values)
plt.title("Métricas de Avaliação do Modelo CNN")
plt.xlabel("Métricas")
plt.ylabel("Valor")
plt.show()

# Definir um contador global
i = 0

=======
>>>>>>> parent of 444c522 (Create modelo_cnn_irrigacao.h5)
# Salvar o modelo
modelo_cnn.save(f"modelo_cnn_irrigacao_{i}.keras") # Adiciona o contador i para salvar com data ou numero
i += 1
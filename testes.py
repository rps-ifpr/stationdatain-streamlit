import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Geração de Dados Sintéticos
def gerar_dados_sinteticos(num_amostras=1000):
    X = np.random.randn (num_amostras, 10)
    pesos = np.random.randn (10, 1)
    vieses = np.random.randn (1)
    Y = np.dot (X, pesos) + vieses
    Y = np.where (Y > 0, 1, 0)
    return X, Y.squeeze ()


# Construção do Modelo
def construir_modelo(dimensao_entrada):
    modelo = Sequential ([
        Dense (64, activation='relu', input_shape=(dimensao_entrada,)),
        Dense (64, activation='relu'),
        Dense (1, activation='sigmoid')
    ])
    modelo.compile (optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo


# Visualização de Métricas
def plotar_metricas(historico):
    plt.figure (figsize=(10, 4))
    plt.subplot (1, 2, 1)
    plt.plot (historico.history['accuracy'], label='Acurácia')
    plt.plot (historico.history['val_accuracy'], label='Acurácia Validação')
    plt.xlabel ('Época')
    plt.ylabel ('Acurácia')
    plt.legend ()

    plt.subplot (1, 2, 2)
    plt.plot (historico.history['loss'], label='Perda')
    plt.plot (historico.history['val_loss'], label='Perda Validação')
    plt.xlabel ('Época')
    plt.ylabel ('Perda')
    plt.legend ()
    plt.tight_layout ()
    st.pyplot (plt)


# Interface Streamlit
st.title ('Treinamento de Modelo e Visualização de Métricas')

if st.button ('Gerar Dados e Treinar Modelo'):
    X, Y = gerar_dados_sinteticos (1000)
    X_treino, X_teste, Y_treino, Y_teste = train_test_split (X, Y, test_size=0.2, random_state=42)

    modelo = construir_modelo (X_treino.shape[1])
    historico = modelo.fit (X_treino, Y_treino, validation_split=0.2, epochs=20, batch_size=32, verbose=1)

    plotar_metricas (historico)

    Y_predito = (modelo.predict (X_teste) > 0.5).astype ("int32")
    cm = confusion_matrix (Y_teste, Y_predito)
    st.write ('Matriz de Confusão:')
    st.dataframe (pd.DataFrame (cm, columns=['Predito 0', 'Predito 1'], index=['Real 0', 'Real 1']))

    relatorio = classification_report (Y_teste, Y_predito, output_dict=True)
    st.write ('Relatório de Classificação:')
    st.dataframe (pd.DataFrame (relatorio).transpose ())


import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Carregar os dados
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Pré-processar os dados
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Construir o modelo
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(Dense(10, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_split=0.2)

# Visualizar a acurácia e a perda
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Acurácia
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Acurácia Treino')
plt.plot(epochs, val_acc, 'b', label='Acurácia Validação')
plt.title('Acurácia de Treinamento e Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Perda
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Perda Treino')
plt.plot(epochs, val_loss, 'b', label='Perda Validação')
plt.title('Perda de Treinamento e Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.show()

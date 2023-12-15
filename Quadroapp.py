import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Função para carregar e processar dados
def load_and_process_data(file_path, delimiter=';'):
    df = pd.read_csv(file_path, delimiter=delimiter)

    # Convertendo variáveis para numéricas
    colunas_para_converter = ['Outdoor Temperature(°C)', 'Outdoor Humidity(%)', 'Wind Speed(km/h)',
                              'Gust(km/h)', 'DewPoint(°C)', 'WindChill(°C)']
    df[colunas_para_converter] = df[colunas_para_converter].apply(pd.to_numeric, errors='coerce')

    # Renomeando colunas
    column_mapping = {
        'n': 'id',
        'Time': 'date',
        'Interval': 'intervalo',
        'Indoor Temperature(°C)': 'internal_temp',
        'Indoor Humidity(%)': 'internal_humidity',
        'Outdoor Temperature(°C)': 'external_temp',
        'Outdoor Humidity(%)': 'external_humidity',
        'Relative Pressure(mmHg)': 'relative_pressure',
        'Absolute Pressure(mmHg)': 'absolute_pressure',
        'Wind Speed(km/h)': 'wind_speed',
        'Gust(km/h)': 'gust_wind',
        'Wind Direction': 'wind_direction',
        'DewPoint(°C)': 'dew_point',
        'WindChill(°C)': 'thermal_sensation',
        'Hour Rainfall(mm)': 'rain_time',
        '24 Hour Rainfall(mm)': 'rain_24h',
        'Week Rainfall(mm)': 'rain_week',
        'Month Rainfall(mm)': 'rain_month',
        'Total Rainfall(mm)': 'total_rain'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Removendo outliers
    colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for coluna in colunas_numericas:
        Q1, Q3 = df[coluna].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        filtro = (df[coluna] >= Q1 - 1.5 * IQR) & (df[coluna] <= Q3 + 1.5 * IQR)
        df = df.loc[filtro]

    # Removendo valores nulos
    df.dropna(inplace=True)

    return df

# Função para plotar a matriz de confusão
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    st.pyplot(fig)

# Título da aplicação Streamlit
st.title("Estação Meteorológica IFPR-Campus Capanema - Estudo de Caso 3")

# Carregando e processando os dados
df = load_and_process_data('dados.csv')

# Selecionando colunas para análise
colunas_analise = ['external_temp', 'external_humidity', 'wind_speed', 'gust_wind', 'dew_point', 'thermal_sensation', 'absolute_pressure']

# Criando dataframe de correlação
corr = df[colunas_analise].corr()

# Plotando o gráfico de correlação
st.write("Gráfico de correlação dos dados sem outliers")
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Adicionando uma coluna de target (por exemplo, chuva ou não chuva) para fins de classificação
df['target'] = np.where(df['rain_time'] > 0, 1, 0)

# Separando os dados em recursos (X) e rótulos (y)
X = df[colunas_analise]
y = df['target']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Machine Learning - Random Forest
st.write("### Machine Learning - Random Forest")

# Criando o modelo
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Fazendo previsões
y_pred_rf = rf_model.predict(X_test)

# Avaliando o modelo
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_report_rf = classification_report(y_test, y_pred_rf)

st.write(f"Acurácia do modelo de Machine Learning (Random Forest): {accuracy_rf:.2f}")
st.write("Relatório de Classificação:")
st.write(classification_report_rf)

# Deep Learning - Rede Neural Simples
st.write("### Deep Learning - Rede Neural Simples")

# Criando o modelo
dl_model = Sequential()
dl_model.add(Dense(64, input_dim=len(colunas_analise), activation='relu'))
dl_model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo
dl_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
dl_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Avaliando o modelo
_, accuracy_dl = dl_model.evaluate(X_test, y_test)
st.write(f"Acurácia do modelo de Deep Learning (Rede Neural Simples): {accuracy_dl:.2f}")

# Plotando a matriz de confusão para o modelo de Machine Learning (Random Forest)
st.write("### Matriz de Confusão - Machine Learning (Random Forest)")
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")

# Plotando a matriz de confusão para o modelo de Deep Learning (Rede Neural Simples)
st.write("### Matriz de Confusão - Deep Learning (Rede Neural Simples)")
y_pred_dl = (dl_model.predict(X_test) > 0.5).astype("int32")  # Convertendo probabilidades para rótulos binários
plot_confusion_matrix(y_test, y_pred_dl, "Deep Learning")

# Adicionando gráficos de barras para comparar acurácia
fig, ax = plt.subplots()
models = ['Machine Learning (Random Forest)', 'Deep Learning (Neural Network)']
accuracy_scores = [accuracy_rf, accuracy_dl]
sns.barplot(x=models, y=accuracy_scores, ax=ax)
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Model Accuracy')
st.pyplot(fig)
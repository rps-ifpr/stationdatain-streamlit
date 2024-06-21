import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt

# Carregue os dados combinados
def carregar_dados(nome_arquivo):
    """Carrega os dados combinados do arquivo CSV."""
    return pd.read_csv(nome_arquivo)

# Pré-processamento dos dados
def pre_processar_dados(dados, variaveis_entrada, variavel_saida):
    """Pré-processa os dados: converte 'estagio_cultura' para numérico e normaliza."""
    label_encoder = LabelEncoder()
    dados['estagio_cultura'] = label_encoder.fit_transform(dados['estagio_cultura'])
    scaler = MinMaxScaler()
    dados[variaveis_entrada] = scaler.fit_transform(dados[variaveis_entrada])
    dados[variavel_saida] = scaler.fit_transform(dados[[variavel_saida]])
    return dados

# Crie o modelo LSTM
def criar_modelo_lstm(configuracao):
    """Cria o modelo LSTM de acordo com a configuração fornecida."""
    input_tensor = Input(shape=(1, 9))
    x = LSTM(configuracao['unidades_lstm'], return_sequences=True, recurrent_regularizer=l2(0.01))(input_tensor)
    x = Dropout(configuracao['dropout'])(x)
    for _ in range(configuracao['n_camadas_lstm'] - 1):
        x = LSTM(configuracao['unidades_lstm'], recurrent_regularizer=l2(0.01))(x)
        x = Dropout(configuracao['dropout'])(x)
    output_tensor = Dense(1)(x)
    modelo = Model(inputs=input_tensor, outputs=output_tensor)
    modelo.compile(loss='mean_squared_error', optimizer=configuracao['otimizador'])
    configuracao['otimizador'].learning_rate = configuracao['taxa_aprendizado']
    return modelo

# Treinamento com validação cruzada
def treinar_modelo(modelo, X_treinamento, y_treinamento, X_teste, y_teste, epochs=100, batch_size=32):
    """Treina o modelo usando validação cruzada."""
    history = modelo.fit(X_treinamento, y_treinamento, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_teste, y_teste), verbose=0)
    return history

# Avaliação do modelo
def avaliar_modelo(modelo, X_teste, y_teste):
    """Avalia o desempenho do modelo e retorna as métricas."""
    y_previsao = modelo.predict(X_teste)
    rmse = np.sqrt(mean_squared_error(y_teste, y_previsao))
    mae = mean_absolute_error(y_teste, y_previsao)
    mape = mean_absolute_percentage_error(y_teste, y_previsao)
    return rmse, mae, mape

# Plotagem da curva de aprendizagem
def plotar_curva_aprendizagem(loss_train, loss_val):
    """Plota a curva de aprendizagem (perda de treinamento e validação)."""
    plt.figure(figsize=(10, 6))
    plt.plot(np.mean(loss_train, axis=0), label='Perda de Treinamento')
    plt.plot(np.mean(loss_val, axis=0), label='Perda de Validação')
    plt.title('Curva de Aprendizagem do Modelo LSTM')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()

# Plotagem da perda para cada fold
def plotar_perda_folds(loss_train_all, loss_val_all):
    """Plota a perda de treinamento e validação para cada fold da validação cruzada."""
    plt.figure(figsize=(10, 6))
    for i in range(len(loss_train_all)):
        plt.plot(loss_train_all[i], label=f'Fold {i+1} - Treinamento')
        plt.plot(loss_val_all[i], label=f'Fold {i+1} - Validação')
    plt.title('Perda ao Longo das Épocas (Validação Cruzada)')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()

# Defina as variáveis de entrada e saída
variaveis_entrada = ['satelite_solo', 'satelite_agua', 'satelite_vegetacao', 'umidade_solo',
                    'condutividade_eletrica', 'temperatura_solo', 'estagio_cultura',
                    'precipitacao_previsao', 'chuva_historica']
variavel_saida = 'irrigacao_previsao'

# Defina as configurações a serem testadas
configuracoes = [
    # Arquitetura 1
    {'n_camadas_lstm': 2, 'unidades_lstm': 50, 'dropout': 0.2, 'otimizador': Adam(), 'taxa_aprendizado': 0.001},
    # Arquitetura 2
    {'n_camadas_lstm': 3, 'unidades_lstm': 100, 'dropout': 0.3, 'otimizador': SGD(), 'taxa_aprendizado': 0.01},
    # Arquitetura 3
    {'n_camadas_lstm': 2, 'unidades_lstm': 25, 'dropout': 0.1, 'otimizador': RMSprop(), 'taxa_aprendizado': 0.005},
    # ... adicione mais configurações ...
]

# Lista para armazenar as métricas de desempenho de cada configuração
rmse_scores = []
mae_scores = []
mape_scores = []

# Lista para armazenar a perda de treinamento e validação para cada configuração
loss_train_all = []
loss_val_all = []

# Carregue os dados
dados_combinados = carregar_dados('dados_combinados.csv')

# Pré-processamento dos dados
dados_combinados = pre_processar_dados(dados_combinados, variaveis_entrada, variavel_saida)

# Treine e avalie o modelo para cada configuração
for configuracao in configuracoes:
    # Crie o modelo LSTM
    modelo_conjunto = criar_modelo_lstm(configuracao)

    # Treinamento com validação cruzada
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores_config = []
    mae_scores_config = []
    mape_scores_config = []
    loss_train = []
    loss_val = []

    for train_index, test_index in tscv.split(dados_combinados):
        X_treinamento, X_teste = dados_combinados[variaveis_entrada].iloc[train_index], dados_combinados[variaveis_entrada].iloc[train_index]
        y_treinamento, y_teste = dados_combinados[variavel_saida].iloc[train_index], dados_combinados[variavel_saida].iloc[train_index]

        # Reorganize os dados
        X_treinamento = X_treinamento.transpose()
        X_teste = X_teste.transpose()

        # Treine o modelo
        history = treinar_modelo(modelo_conjunto, X_treinamento, y_treinamento, X_teste, y_teste)

        # Avalie o desempenho do modelo
        rmse, mae, mape = avaliar_modelo(modelo_conjunto, X_teste, y_teste)

        rmse_scores_config.append(rmse)
        mae_scores_config.append(mae)
        mape_scores_config.append(mape)

        # Armazene a perda de treinamento e validação para cada fold
        loss_train.append(history.history['loss'])
        loss_val.append(history.history['val_loss'])

    rmse_scores.append(np.mean(rmse_scores_config))
    mae_scores.append(np.mean(mae_scores_config))
    mape_scores.append(np.mean(mape_scores_config))
    loss_train_all.append(loss_train)
    loss_val_all.append(loss_val)

    print(f'Configuração: {configuracao}')
    print(f'RMSE: {rmse_scores[-1]}')
    print(f'MAE: {mae_scores[-1]}')
    print(f'MAPE: {mape_scores[-1]}')

# Encontre a melhor configuração
melhor_indice = np.argmin(rmse_scores)
melhor_configuracao = configuracoes[melhor_indice]

# Imprima a melhor configuração e suas métricas
print(f'\nMelhor configuração: {melhor_configuracao}')
print(f'RMSE: {rmse_scores[melhor_indice]}')
print(f'MAE: {mae_scores[melhor_indice]}')
print(f'MAPE: {mape_scores[melhor_indice]}')

# Plota a curva de aprendizagem da melhor configuração
plotar_curva_aprendizagem(loss_train_all[melhor_indice], loss_val_all[melhor_indice])

# Plota a perda ao longo das épocas para cada fold da melhor configuração
plotar_perda_folds(loss_train_all[melhor_indice], loss_val_all[melhor_indice])

# Salve o modelo treinado
modelo_conjunto.save('modelo_conjunto1.h5')
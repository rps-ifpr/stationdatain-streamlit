import pandas as pd
import streamlit as st
import time
import random
import requests


def carregar_previsoes(nome_arquivo):
    """Carrega as previsões de um modelo a partir de um arquivo CSV."""
    try:
        df = pd.read_csv (nome_arquivo)
        return df
    except FileNotFoundError:
        st.error (f"Arquivo '{nome_arquivo}' não encontrado.")
        return None


def combinar_previsoes(previsoes_satelite, previsoes_sensores, previsoes_cultura, previsoes_estacao,
                       previsoes_temporais):
    """Combina as previsões dos modelos em um único DataFrame."""
    previsoes_combinadas = pd.concat ([
        previsoes_satelite,
        previsoes_sensores,
        previsoes_cultura,
        previsoes_estacao,
        previsoes_temporais
    ], axis=1)
    return previsoes_combinadas


def calcular_previsao_final(previsoes_combinadas, estagio, pesos):
    """Calcula a previsão final de irrigação com base no estágio da planta."""
    # Cria uma coluna para o estágio de desenvolvimento
    previsoes_combinadas['estagio_cultura'] = previsoes_combinadas['estagio_cultura'].astype ('category')

    # Calcula a previsão final utilizando a média ponderada
    previsoes_combinadas['previsao_final'] = 0
    for modelo, peso in pesos[estagio].items ():
        previsoes_combinadas['previsao_final'] += previsoes_combinadas[modelo] * peso

    return previsoes_combinadas


def obter_quantidade_agua(estagio, tipo_irrigacao, tipo_solo):
    """Retorna a quantidade de água a ser aplicada com base na tabela de irrigação."""
    tabela_irrigacao = {
        'Vegetativo': {
            'Aspersão': {
                'Franco-arenoso': 50,
                'Argiloso': 40,
                'Areia': 60
            },
            'Gotejamento': {
                'Franco-arenoso': 30,
                'Argiloso': 25,
                'Areia': 40
            }
        },
        'Floração': {
            'Aspersão': {
                'Franco-arenoso': 40,
                'Argiloso': 35,
                'Areia': 50
            },
            'Gotejamento': {
                'Franco-arenoso': 25,
                'Argiloso': 20,
                'Areia': 30
            }
        },
        'Colheita': {
            'Aspersão': {
                'Franco-arenoso': 30,
                'Argiloso': 25,
                'Areia': 40
            },
            'Gotejamento': {
                'Franco-arenoso': 15,
                'Argiloso': 10,
                'Areia': 20
            }
        }
    }
    return tabela_irrigacao[estagio][tipo_irrigacao][tipo_solo]


def get_weather_data(city):
    """Obtém dados climáticos da API OpenWeatherMap."""
    api_key = "342e60213afbec5b9b52e7f87d90248e"  # Substitua pela sua chave API do OpenWeatherMap
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city
    response = requests.get (complete_url)
    if response.status_code == 200:
        data = response.json ()
        main = data['main']
        temp = round ((main['temp'] - 273.15), 1)  # Convertendo de Kelvin para Celsius
        humidity = main['humidity']
        return temp, humidity
    else:
        return None, None


def main():
    st.title ("Sys-Previsão de Irrigação Lúpulo")

    # --- Barra Lateral ---
    with st.sidebar:
        st.subheader ("Escolha as Opções:")
        estagio = st.selectbox ("Selecione o estágio da planta:", ['Vegetativo', 'Floração', 'Colheita'])
        tipo_irrigacao = st.selectbox ("Selecione o tipo de irrigação:", ['Aspersão', 'Gotejamento'])
        tipo_solo = st.selectbox ("Selecione o tipo de solo:", ['Franco-arenoso', 'Argiloso', 'Areia'])

    # --- Conteúdo Principal ---
    # Dados climáticos em tempo real
    city = "Capanema, BR"  # Substitua pela cidade desejada
    temp, humidity = get_weather_data (city)
    st.markdown ("----------")
    # Exibe os dados climáticos em um layout de matriz
    st.markdown (
        f"""
        | **Dados Climáticos** | **Valor** |
        |---|---|
        | Temperatura do Ar | {temp}°C |
        | Umidade do Ar | {humidity}% |
        """
    )
    st.markdown ("----------")
    # Simula um sensor de temperatura em tempo real (com valores aleatórios)
    st.markdown (
        f"""
        | **Dados do Solo** | **Valor** |
        |---|---|
        | Temperatura do Solo (Tempo Real) | {round (random.uniform (15, 30), 1)}°C |
        """
    )

    # Título do Sistema
    st.markdown ("----------")

    # Carregue as previsões de cada modelo
    previsoes_satelite = carregar_previsoes ('previsoes_satelite.csv')
    previsoes_sensores = carregar_previsoes ('previsoes_sensores.csv')
    previsoes_cultura = carregar_previsoes ('previsoes_cultura.csv')
    previsoes_estacao = carregar_previsoes ('previsoes_estacao.csv')
    previsoes_temporais = carregar_previsoes ('previsoes_temporais.csv')

    if all ([not df.empty for df in
             [previsoes_satelite, previsoes_sensores, previsoes_cultura, previsoes_estacao, previsoes_temporais]]):
        # Combine as previsões
        previsoes_combinadas = combinar_previsoes (previsoes_satelite, previsoes_sensores, previsoes_cultura,
                                                   previsoes_estacao, previsoes_temporais)

        # Define os pesos para cada modelo em cada fase de desenvolvimento
        pesos = {
            'Vegetativo': {
                'satelite_solo': 0.1,
                'satelite_agua': 0.1,
                'satelite_vegetacao': 0.1,
                'umidade_solo': 0.3,
                'condutividade_eletrica': 0.3,
                'temperatura_solo': 0.3,
                'cultura_demanda': 0.5,
                'estacao_precipitacao': 0.1,
                'temporais_chuva': 0.0
            },
            'Floração': {
                'satelite_solo': 0.1,
                'satelite_agua': 0.1,
                'satelite_vegetacao': 0.1,
                'umidade_solo': 0.2,
                'condutividade_eletrica': 0.2,
                'temperatura_solo': 0.2,
                'cultura_demanda': 0.6,
                'estacao_precipitacao': 0.1,
                'temporais_chuva': 0.0
            },
            'Colheita': {
                'satelite_solo': 0.1,
                'satelite_agua': 0.1,
                'satelite_vegetacao': 0.1,
                'umidade_solo': 0.1,
                'condutividade_eletrica': 0.1,
                'temperatura_solo': 0.1,
                'cultura_demanda': 0.7,
                'estacao_precipitacao': 0.1,
                'temporais_chuva': 0.0
            }
        }

        # Calcula a previsão final
        previsoes_combinadas = calcular_previsao_final (previsoes_combinadas, estagio, pesos)

        # Obtém a quantidade de água da tabela de irrigação
        quantidade_agua = obter_quantidade_agua (estagio, tipo_irrigacao, tipo_solo)

        # Exibe o resultado
        previsao_final = previsoes_combinadas[
            'previsao_final'].mean ()  # Calcula a média das previsões para o estagio selecionado
        if previsao_final >= 0.5:
            st.success (
                f"Irrigar! A previsão de irrigação para o estágio {estagio} é alta: {previsao_final:.2f}. Aplique {quantidade_agua} litros de água por m².")
        else:
            st.info (f"Não irrigar! A previsão de irrigação para o estágio {estagio} é baixa: {previsao_final:.2f}.")
    else:
        st.error ("Erro ao carregar os arquivos de previsão.")


if __name__ == "__main__":
    main ()
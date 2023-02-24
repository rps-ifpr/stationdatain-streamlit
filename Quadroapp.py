import pandas as pd

# lendo o arquivo csv
df = pd.read_csv('dados.csv', delimiter=';')

# renomeando as colunas para facilitar a manipulação
df.columns = ['n', 'Time', 'Interval', 'Indoor_Temperature', 'Indoor_Humidity',
              'Outdoor_Temperature', 'Outdoor_Humidity', 'Relative_Pressure',
              'Absolute_Pressure', 'Wind_Speed', 'Gust', 'Wind_Direction',
              'DewPoint', 'WindChill', 'Hour_Rainfall', '24_Hour_Rainfall',
              'Week_Rainfall', 'Month_Rainfall', 'Total_Rainfall']

# removendo colunas desnecessárias
cols_to_drop = ['n', 'Time', 'Interval']
df.drop(cols_to_drop, axis=1, inplace=True)

# removendo linhas com valores nulos
df.dropna(inplace=True)

# convertendo colunas para o tipo correto
numeric_cols = ['Indoor_Temperature', 'Indoor_Humidity', 'Outdoor_Temperature',
                'Outdoor_Humidity', 'Relative_Pressure', 'Absolute_Pressure',
                'Wind_Speed', 'Gust', 'DewPoint', 'WindChill', 'Hour_Rainfall',
                '24_Hour_Rainfall', 'Week_Rainfall', 'Month_Rainfall', 'Total_Rainfall']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# removendo valores extremos
df = df[(df['Indoor_Temperature'] > -20) & (df['Indoor_Temperature'] < 60)]
df = df[(df['Indoor_Humidity'] > 0) & (df['Indoor_Humidity'] < 100)]
df = df[(df['Outdoor_Temperature'] > -50) & (df['Outdoor_Temperature'] < 70)]
df = df[(df['Outdoor_Humidity'] > 0) & (df['Outdoor_Humidity'] < 100)]
df = df[(df['Relative_Pressure'] > 500) & (df['Relative_Pressure'] < 1200)]
df = df[(df['Absolute_Pressure'] > 500) & (df['Absolute_Pressure'] < 1200)]
df = df[(df['Wind_Speed'] > 0) & (df['Wind_Speed'] < 200)]
df = df[(df['Gust'] > 0) & (df['Gust'] < 250)]
df = df[(df['DewPoint'] > -40) & (df['DewPoint'] < 50)]
df = df[(df['WindChill'] > -60) & (df['WindChill'] < 60)]
df = df[(df['Hour_Rainfall'] >= 0) & (df['Hour_Rainfall'] < 100)]
df = df[(df['24_Hour_Rainfall'] >= 0) & (df['24_Hour_Rainfall'] < 500)]
df = df[(df['Week_Rainfall'] >= 0) & (df['Week_Rainfall'] < 1000)]
df = df[(df['Month_Rainfall'] >= 0) & (df['Month_Rainfall'] < 5000)]
df = df[(df['Total_Rainfall'] >= 0) & (df['Total_Rainfall'] < 10000)]

#

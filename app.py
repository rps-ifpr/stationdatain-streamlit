import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# importando os dados
df= pd.read_csv ('dados.csv', delimiter=';')

st.title("Dados da estação meteorológica")
st.write("Aqui estão os dados de estação Local:")

first_10_rows = df.head(10)
st.write(first_10_rows)






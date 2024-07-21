import streamlit as st
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.title("Árvore de Decisão")

# Define as características e as classes
features = ['Development Phase', 'Soil Type', 'Irrigation Type']
classes = ['Low', 'Moderate', 'Moderate to High', 'High']

# Define as regras da árvore de decisão
decision_rules = [
    ['Inicial', 'Franco-arenoso', 'Aspersão', 'Baixa'],
    ['Inicial', 'Argiloso', 'Aspersão', 'Moderada'],
    ['Inicial', 'Areia', 'Aspersão', 'Baixa'],
    ['Inicial', 'Franco-arenoso', 'Gotejamento', 'Moderada'],
    ['Inicial', 'Argiloso', 'Gotejamento', 'Moderada a Alta'],
    ['Inicial', 'Areia', 'Gotejamento', 'Moderada'],
    ['Crescimento Rápido', 'Franco-arenoso', 'Aspersão', 'Moderada a Alta'],
    ['Crescimento Rápido', 'Argiloso', 'Aspersão', 'Alta'],
    ['Crescimento Rápido', 'Areia', 'Aspersão', 'Moderada a Alta'],
    ['Crescimento Rápido', 'Franco-arenoso', 'Gotejamento', 'Alta'],
    ['Crescimento Rápido', 'Argiloso', 'Gotejamento', 'Alta'],
    ['Crescimento Rápido', 'Areia', 'Gotejamento', 'Alta'],
    ['Crescimento Médio', 'Franco-arenoso', 'Aspersão', 'Alta'],
    ['Crescimento Médio', 'Argiloso', 'Aspersão', 'Alta'],
    ['Crescimento Médio', 'Areia', 'Aspersão', 'Alta'],
    ['Crescimento Médio', 'Franco-arenoso', 'Gotejamento', 'Alta'],
    ['Crescimento Médio', 'Argiloso', 'Gotejamento', 'Alta'],
    ['Crescimento Médio', 'Areia', 'Gotejamento', 'Alta'],
    ['Final', 'Franco-arenoso', 'Aspersão', 'Moderada'],
    ['Final', 'Argiloso', 'Aspersão', 'Moderada'],
    ['Final', 'Areia', 'Aspersão', 'Moderada'],
    ['Final', 'Franco-arenoso', 'Gotejamento', 'Moderada'],
    ['Final', 'Argiloso', 'Gotejamento', 'Moderada'],
    ['Final', 'Areia', 'Gotejamento', 'Moderada'],
]

# Crie um modelo de árvore de decisão
decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Treina o modelo (neste caso, o treinamento é irrelevante para o exemplo)
decision_tree.fit([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]], [0, 1, 2, 3, 3, 3])

# Plota a árvore de decisão
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(decision_tree, feature_names=features, class_names=classes, filled=True)
plt.title("Decision tree")
st.pyplot(fig)
import tensorflow as tf
from tensorflow.keras import layers, models
def create_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu')
    ])
    return model

def create_lstm(input_shape):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(32),
        layers.Dense(32, activation='relu')
    ])
    return model

def create_dnn(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu')
    ])
    return model

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Supondo que `inputs` seja uma lista de tensores [h_sat, h_temp, h_meteo, h_solo, h_cult]
        # Concatena as entradas
        x = tf.concat(inputs, axis=-1)
        # Aplica atenção
        attention_scores = tf.nn.softmax(layers.Dense(1)(x), axis=1)
        weighted_output = x * attention_scores
        return tf.reduce_sum(weighted_output, axis=1)
def create_final_model(input_shapes):
    # Cria as sub-redes
    cnn_input = layers.Input(shape=input_shapes['satellite'])
    lstm_input = layers.Input(shape=input_shapes['temporal'])
    dnn_input = layers.Input(shape=input_shapes['structured'])

    cnn = create_cnn(input_shapes['satellite'])(cnn_input)
    lstm = create_lstm(input_shapes['temporal'])(lstm_input)
    dnn = create_dnn(input_shapes['structured'])(dnn_input)

    # Fusão com atenção
    fused = AttentionLayer()([cnn, lstm, dnn])

    # Camada de decisão
    decision_output = layers.Dense(1, activation='sigmoid')(fused)

    model = models.Model(inputs=[cnn_input, lstm_input, dnn_input], outputs=decision_output)

    return model

model = create_final_model({
    'satellite': (64, 64, 3),  # Exemplo de shape para imagens de satélite
    'temporal': (120, 10),     # Exemplo de shape para séries temporais (120 timesteps, 10 features)
    'structured': (10,)        # Exemplo de shape para dados estruturados (10 features)
})

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(...) # Adicione aqui o treinamento com seus dado
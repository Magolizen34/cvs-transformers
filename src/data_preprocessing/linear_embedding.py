import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

class LinearEmbedding:
    def __init__(self, input_dim, embedding_dim):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        embedding_layer = Dense(self.embedding_dim, activation='linear')(input_layer)

        model = Model(inputs=input_layer, outputs=embedding_layer)
        model.compile(optimizer='adam', loss='mean_squared_error')

        return model

    def fit_transform(self, numerical_data):
        self.model.fit(numerical_data, numerical_data, epochs=10, verbose=0)
        return self.model.predict(numerical_data)
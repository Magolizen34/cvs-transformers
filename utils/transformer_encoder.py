import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Embedding, GlobalAveragePooling1D

class TransformerEncoder:
    def __init__(self, input_dim, num_heads, embedding_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        embedding_layer = Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim)(input_layer)
        attention_layer = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embedding_dim)(embedding_layer, embedding_layer)
        pooling_layer = GlobalAveragePooling1D()(attention_layer)
        hidden_layer = Dense(self.hidden_dim, activation='relu')(pooling_layer)
        output_layer = Dense(self.output_dim, activation='softmax')(hidden_layer)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model

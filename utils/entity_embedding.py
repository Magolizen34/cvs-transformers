from sklearn.preprocessing import OneHotEncoder
import numpy as np

class EntityEmbedding:
    def __init__(self, categories, embedding_dim):
        self.categories = categories
        self.embedding_dim = embedding_dim
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.embeddings = None

    def fit_transform(self, categorical_data):
        encoded_data = self.encoder.fit_transform(categorical_data.reshape(-1, 1)).toarray()
        self.embeddings = np.random.randn(len(self.categories), self.embedding_dim)
        return self.embeddings[np.argmax(encoded_data, axis=1)]
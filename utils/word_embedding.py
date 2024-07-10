from gensim.models import Word2Vec
import numpy as np

class WordEmbedding:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.model = Word2Vec(size=embedding_dim, window=5, min_count=1, workers=4)

    def fit_transform(self, categorical_data):
        sentences = [str(item) for item in categorical_data]
        tokenized_data = [sentence.split() for sentence in sentences]
        self.model.build_vocab(tokenized_data)
        self.model.train(tokenized_data, total_examples=len(tokenized_data), epochs=10)
        return np.array([self.model.wv[str(item)] for item in categorical_data])

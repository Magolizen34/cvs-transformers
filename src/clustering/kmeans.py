from sklearn.cluster import KMeans

class KMeansClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters)

    def fit_predict(self, data):
        return self.model.fit_predict(data)

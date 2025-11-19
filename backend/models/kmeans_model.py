import pickle
from sklearn.cluster import KMeans


class KMeansModel:
    """
    Thin wrapper over scikit-learn KMeans with a simple API used by the pipeline.
    Provides train/predict/save/load methods for consistency with other models.
    """
    def __init__(self, n_clusters=5):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

    def train(self, X):
        """
        Fits the KMeans model on features X (n_samples x n_features).
        """
        self.model.fit(X)
        return self.model

    def predict(self, X):
        """
        Returns cluster labels for each row in X.
        """
        return self.model.predict(X)

    def save_model(self, path="data/model_files/kmeans_model.pkl"):
        """
        Serializes the fitted KMeans model to disk.
        """
        pickle.dump(self.model, open(path, "wb"))

    def load_model(self, path="data/model_files/kmeans_model.pkl"):
        """
        Loads a serialized KMeans model from disk into this instance.
        """
        self.model = pickle.load(open(path, "rb"))



import pickle
from sklearn.linear_model import LogisticRegression


class LogisticModel:
    """
    Thin wrapper around scikit-learn LogisticRegression providing
    a consistent API (train/predict/save/load) for this project.
    The model predicts match probability for feature vectors.
    """
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, X, y):
        """
        Fits the logistic regression model.
        X: feature matrix
        y: binary labels (0/1)
        """
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """
        Returns probability of the positive class for each row in X.
        """
        return self.model.predict_proba(X)[:, 1]  # match probability

    def save_model(self, path="data/model_files/logistic_model.pkl"):
        """
        Serializes the underlying sklearn model to disk.
        """
        pickle.dump(self.model, open(path, "wb"))

    def load_model(self, path="data/model_files/logistic_model.pkl"):
        """
        Loads the serialized sklearn model from disk into this instance.
        """
        self.model = pickle.load(open(path, "rb"))




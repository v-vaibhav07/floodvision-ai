
import numpy as np
from sklearn.metrics import pairwise_distances_argmin


class FrozenKMeans:

    def __init__(self, centers):
        self.cluster_centers_ = np.asarray(
            centers,
            dtype=np.float32
        ).copy()

        self.n_clusters = len(self.cluster_centers_)
        self.n_features_in_ = self.cluster_centers_.shape[1]
        self.n_iter_ = 0

    def predict(self, X):
        return pairwise_distances_argmin(
            np.asarray(X, dtype=np.float32),
            self.cluster_centers_
        )

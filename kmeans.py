import numpy as np


class KMeans:
    """
    K-Means clustering algorithm implemented from scratch using NumPy.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K).
    max_iter : int
        Maximum number of iterations before stopping.
    random_state : int or None
        Seed for reproducibility. None means random each run.
    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 300, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

        # Set after fit()
        self.X_fit_ = None
        self.labels_ = None
        self.centroids = None

        # Full iteration history (useful for visualization and debugging)
        self.labels_history: list[np.ndarray] = []
        self.centroids_history: list[np.ndarray] = []
        self.costs: list[float] = []

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-Means on dataset X."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Reset history in case fit() is called multiple times
        self.labels_history = []
        self.centroids_history = []
        self.costs = []

        self.X_fit_ = X.copy()

        # Initialize centroids by randomly sampling K points from X
        init_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[init_indices].copy()
        self.centroids_history.append(self.centroids.copy())

        for _ in range(self.max_iter):
            labels = self._get_labels(X)
            self.labels_history.append(labels.copy())

            new_centroids = self._get_centroids(X, labels)
            self.centroids = new_centroids
            self.centroids_history.append(new_centroids.copy())

            cost = self._calculate_cost(X)
            self.costs.append(cost)

            # Converged when centroids stop moving
            if np.allclose(self.centroids_history[-1], self.centroids_history[-2]):
                break

        self.labels_ = self.labels_history[-1]
        self.centroids = self.centroids_history[-1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each point in X to its nearest centroid."""
        return self._get_labels(X)

    def _get_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances from every point to every centroid.

        Returns
        -------
        distances : np.ndarray, shape (n_samples, n_clusters)
        """
        diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]  # (n, k, d)
        return np.sum(diff ** 2, axis=2)  # (n, k)

    def _get_labels(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid index."""
        return np.argmin(self._get_distances(X), axis=1)

    def _get_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recompute centroids as the mean of all points assigned to each cluster."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
            else:
                # Empty cluster: reinitialize randomly to avoid dead centroid
                centroids[k] = X[np.random.choice(X.shape[0])]
        return centroids

    def _calculate_cost(self, X: np.ndarray) -> float:
        """Within-cluster sum of squares (WCSS)."""
        labels = self._get_labels(X)
        distances = self._get_distances(X)
        return float(distances[np.arange(X.shape[0]), labels].sum())

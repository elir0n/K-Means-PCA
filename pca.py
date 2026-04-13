import numpy as np


class PCA:
    """
    Principal Component Analysis implemented from scratch using NumPy.

    Performs eigendecomposition of the covariance matrix to find the
    principal axes of variation in the data.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None       # Per-feature mean used to center data
        self.V: np.ndarray | None = None            # Eigenvectors, columns sorted by descending eigenvalue
        self.eigenvalues_: np.ndarray | None = None # Sorted eigenvalues (descending)

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Fit PCA on dataset X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        """
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix: shape (n_features, n_features)
        cov = np.cov(X_centered, rowvar=False)

        # eigh is numerically stable for real symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort by descending eigenvalue
        idx = np.argsort(eigvals)[::-1]
        self.eigenvalues_ = eigvals[idx]
        self.V = eigvecs[:, idx]  # columns are principal axes

        return self

    def transform(self, X: np.ndarray, n_dimensions: int) -> np.ndarray:
        """
        Project X onto the top-n_dimensions principal components.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        n_dimensions : int

        Returns
        -------
        X_reduced : np.ndarray, shape (n_samples, n_dimensions)
        """
        X_centered = X - self.mean_
        W = self.V[:, :n_dimensions]   # (n_features, n_dimensions)
        return X_centered @ W


def pca_reconstruction(x: np.ndarray, pca: PCA, n_dimensions: int) -> np.ndarray:
    """
    Reconstruct a single sample x from its low-dimensional PCA projection.

    Parameters
    ----------
    x : np.ndarray, shape (n_features,)
    pca : fitted PCA instance
    n_dimensions : int
        Number of principal components to use for reconstruction.

    Returns
    -------
    x_reconstructed : np.ndarray, shape (n_features,)
        Values clamped to [0, 255] for pixel data.
    """
    x_centered = x - pca.mean_
    W = pca.V[:, :n_dimensions]         # (n_features, n_dimensions)
    z = W.T @ x_centered                # project to low-dim space
    x_rec = W @ z + pca.mean_           # reconstruct in original space
    return np.clip(x_rec, 0, 255)

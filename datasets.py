import numpy as np
from sklearn.datasets import fetch_openml, make_moons


def make_convex_blobs(
    n_per_cluster: int = 500,
    centers: list[list[float]] | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Generate a 2D dataset of Gaussian blobs arranged at convex positions.

    Parameters
    ----------
    n_per_cluster : int
        Number of points per cluster.
    centers : list of [x, y] positions, or None
        Defaults to [[0,0], [5,5], [5,0], [0,5]].
    random_state : int or None

    Returns
    -------
    X : np.ndarray, shape (n_per_cluster * n_clusters, 2)
        Shuffled dataset (no labels — this is for unsupervised learning).
    """
    if random_state is not None:
        np.random.seed(random_state)

    if centers is None:
        centers = [[0, 0], [5, 5], [5, 0], [0, 5]]

    clusters = [np.random.normal(loc=c, size=(n_per_cluster, 2)) for c in centers]
    X = np.concatenate(clusters, axis=0)
    np.random.shuffle(X)
    return X


def make_non_convex_moons(
    n_samples: int = 300,
    noise: float = 0.05,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the classic two-moons non-convex dataset.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 2)
    y : np.ndarray, shape (n_samples,) — ground-truth cluster labels (0 or 1)
    """
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the MNIST handwritten digits dataset via OpenML.

    Downloads on first call (~55 MB), cached locally afterwards.

    Returns
    -------
    X : np.ndarray, shape (70000, 784) — pixel values in [0, 255]
    y : np.ndarray, shape (70000,)      — integer labels 0–9
    """
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    y = y.astype(int)
    return X, y

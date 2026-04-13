from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kmeans import KMeans
    from pca import PCA


# ---------------------------------------------------------------------------
# K-Means visualizations
# ---------------------------------------------------------------------------

def plot_dataset(X: np.ndarray, title: str, figsize: tuple = (6, 5)) -> None:
    """Scatter plot of a raw 2D dataset."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(X[:, 0], X[:, 1], s=20, alpha=0.6, edgecolors="none")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_kmeans_progress(
    data: np.ndarray,
    kmeans_model: "KMeans",
    max_steps: int = 5,
) -> None:
    """
    Show the K-Means clustering progress as a single multi-panel figure.

    Each panel corresponds to one iteration, displaying the cluster
    assignments (colored points) and centroids (gold stars).

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, 2)
    kmeans_model : fitted KMeans instance
    max_steps : int
        How many iteration panels to display.
    """
    n_iters = len(kmeans_model.labels_history)
    steps = min(max_steps, n_iters)
    cmap = plt.cm.get_cmap("tab10", kmeans_model.n_clusters)

    fig, axes = plt.subplots(1, steps, figsize=(4 * steps, 4))
    if steps == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        labels = kmeans_model.labels_history[i]
        centroids = kmeans_model.centroids_history[i + 1]  # centroid after the assignment

        for cid in range(kmeans_model.n_clusters):
            pts = data[labels == cid]
            ax.scatter(pts[:, 0], pts[:, 1], color=cmap(cid), alpha=0.7, s=25, edgecolors="none")

        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            marker="*", color="gold", edgecolors="black", linewidths=0.5,
            s=200, zorder=5, label="Centroids",
        )
        ax.set_title(f"Iteration {i + 1}", fontsize=11)
        ax.set_xlabel("Feature 1", fontsize=9)
        ax.set_ylabel("Feature 2", fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)

    fig.suptitle("K-Means Clustering Progress", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cost_curve(costs: list[float], title: str = "K-Means Cost Convergence") -> None:
    """Plot the within-cluster sum of squares (WCSS) over iterations."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(costs) + 1), costs, marker="o", linewidth=2, markersize=5)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("WCSS (Cost)")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# PCA visualizations
# ---------------------------------------------------------------------------

def plot_pca_scatter(
    X_reduced: np.ndarray,
    y: np.ndarray,
    title: str = "PCA: Top 2 Principal Components",
) -> None:
    """
    Scatter plot of data projected onto the first two principal components,
    colored by class label.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="tab10", s=5, alpha=0.5)
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit label")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.tight_layout()
    plt.show()


def plot_projection_matrices(V: np.ndarray, r: int = 5) -> None:
    """
    Visualize the projection matrices V^T V and V V^T for the top-r
    principal components as heatmaps.

    V^T V ≈ I  (orthonormality of principal axes)
    V V^T      (projection / reconstruction operator in original space)
    """
    W = V[:, :r]
    VtV = W.T @ W   # (r, r)
    VVt = W @ W.T   # (d, d)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im1 = axes[0].imshow(VtV, cmap="plasma", aspect="equal")
    axes[0].set_title(r"$V^T V$  (should be ≈ I)", fontsize=12)
    axes[0].set_xlabel("Principal Components")
    axes[0].set_ylabel("Principal Components")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(VVt, cmap="coolwarm", aspect="equal")
    axes[1].set_title(r"$V V^T$  (projection operator)", fontsize=12)
    axes[1].set_xlabel("Data Dimensions")
    axes[1].set_ylabel("Data Dimensions")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Projection matrices — top {r} principal components", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_pca_reconstruction(
    x_orig: np.ndarray,
    pca: "PCA",
    dims_list: list[int] = [3, 10, 100],
) -> None:
    """
    Plot an original MNIST digit alongside its PCA reconstructions at
    various numbers of retained dimensions.

    Parameters
    ----------
    x_orig : np.ndarray, shape (784,)
    pca : fitted PCA instance
    dims_list : list of int
        Number of dimensions to reconstruct with.
    """
    from pca import pca_reconstruction

    n_panels = 1 + len(dims_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))

    axes[0].imshow(x_orig.reshape(28, 28), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for ax, d in zip(axes[1:], dims_list):
        x_rec = pca_reconstruction(x_orig, pca, d)
        ax.imshow(x_rec.reshape(28, 28), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{d} dims", fontsize=11)
        ax.axis("off")

    fig.suptitle("PCA Reconstruction Quality", fontsize=13)
    plt.tight_layout()
    plt.show()

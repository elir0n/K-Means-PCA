"""
main.py — end-to-end demo for the K-Means & PCA mini project.

Run with:
    python main.py

Part 1 runs first (fast), then Part 2 which loads MNIST (slow on first run).
"""

import numpy as np

import datasets
import visualization
from kmeans import KMeans
from pca import PCA


# ---------------------------------------------------------------------------
# Part 1: K-Means Clustering
# ---------------------------------------------------------------------------

def run_part1_kmeans() -> None:
    print("=" * 60)
    print("  Part 1: K-Means Clustering")
    print("=" * 60)

    # --- Data generation ---
    print("\n[1/4] Generating datasets...")
    X_convex = datasets.make_convex_blobs(n_per_cluster=500, random_state=0)
    X_moons, y_moons = datasets.make_non_convex_moons(n_samples=300, noise=0.05, random_state=42)
    print(f"  Convex blobs : {X_convex.shape}")
    print(f"  Two moons    : {X_moons.shape}")

    visualization.plot_dataset(X_convex, "Convex Dataset — 4 Gaussian Blobs")
    visualization.plot_dataset(X_moons, "Non-Convex Dataset — Two Moons")

    # --- K-Means on convex data ---
    print("\n[2/4] Running K-Means on convex data (k=4)...")
    km_convex = KMeans(n_clusters=4, max_iter=300, random_state=42)
    km_convex.fit(X_convex)
    print(f"  Converged in {len(km_convex.costs)} iterations  |  final WCSS = {km_convex.costs[-1]:.1f}")

    visualization.plot_kmeans_progress(X_convex, km_convex, max_steps=5)
    visualization.plot_cost_curve(km_convex.costs, "Convex Data — WCSS Convergence")

    # --- K-Means on non-convex data ---
    print("\n[3/4] Running K-Means on two-moons data (k=2)...")
    km_moons = KMeans(n_clusters=2, max_iter=300, random_state=42)
    km_moons.fit(X_moons)
    print(f"  Converged in {len(km_moons.costs)} iterations  |  final WCSS = {km_moons.costs[-1]:.4f}")
    print("  (K-Means struggles here — the clusters are not convex)")

    visualization.plot_kmeans_progress(X_moons, km_moons, max_steps=5)
    visualization.plot_cost_curve(km_moons.costs, "Non-Convex Data (Moons) — WCSS Convergence")

    print("\n[4/4] Part 1 complete.\n")


# ---------------------------------------------------------------------------
# Part 2: PCA on MNIST
# ---------------------------------------------------------------------------

def run_part2_pca() -> None:
    print("=" * 60)
    print("  Part 2: PCA Dimensionality Reduction on MNIST")
    print("=" * 60)

    # --- Load MNIST ---
    print("\n[1/4] Loading MNIST (downloads ~55 MB on first run)...")
    X, y = datasets.load_mnist()
    print(f"  X shape: {X.shape}  |  classes: {np.unique(y)}")

    # --- Fit PCA ---
    print("\n[2/4] Fitting PCA on MNIST (covariance + eigendecomposition)...")
    pca = PCA()
    pca.fit(X)
    print("  Done.")

    # Project to 2D and visualize
    X_2d = pca.transform(X, n_dimensions=2)
    visualization.plot_pca_scatter(X_2d, y, title="MNIST in 2D PCA Space")

    # --- Projection matrices ---
    print("\n[3/4] Visualizing projection matrices (top 5 components)...")
    visualization.plot_projection_matrices(pca.V, r=5)

    # --- Reconstruction ---
    print("\n[4/4] Showing PCA reconstruction at various dimensions...")
    np.random.seed(7)
    idx = np.random.randint(0, X.shape[0])
    x_orig = X[idx]
    digit_label = y[idx]
    print(f"  Selected sample index {idx}  (digit: {digit_label})")

    visualization.plot_pca_reconstruction(x_orig, pca, dims_list=[3, 10, 100])

    print("\nPart 2 complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_part1_kmeans()
    run_part2_pca()

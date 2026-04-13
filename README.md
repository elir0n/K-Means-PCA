# K-Means & PCA from Scratch

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-pure%20numpy-013243?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

A clean, runnable Python project implementing **K-Means clustering** and **Principal Component Analysis (PCA)** from scratch using only NumPy — no scikit-learn for the core algorithms. Visualizes convergence, cluster structure, and MNIST digit reconstruction across multiple dimensionalities.

---

## Features

- **Pure NumPy K-Means** — random initialization, convergence detection, WCSS cost tracking, and full iteration history for visualization
- **Pure NumPy PCA** — covariance-matrix eigendecomposition, dimensionality reduction, and image reconstruction
- **Convex vs. non-convex clustering** — demonstrates where K-Means succeeds and where it fails geometrically
- **MNIST in 2D** — projects 70,000 × 784-dimensional images to 2D, with visible digit clustering
- **Reconstruction quality** — side-by-side comparison of original and reconstructed digits at 3, 10, and 100 dimensions
- **Projection matrix visualization** — heatmaps of V^T V and V V^T explaining the PCA subspace
- Reproducible with `random_state` parameters throughout

---

## Project Structure

```
K-Means-PCA/
│
├── kmeans.py        # KMeans class — fit, predict, cost tracking, history
├── pca.py           # PCA class — fit, transform; pca_reconstruction() function
├── datasets.py      # make_convex_blobs, make_non_convex_moons, load_mnist
├── visualization.py # All plotting helpers (6 functions)
├── main.py          # End-to-end orchestrator — run this
└── README.md
```

---

## Requirements

```
numpy
matplotlib
scikit-learn   # used only for fetch_openml (MNIST) and make_moons
```

Install everything with:

```bash
pip install numpy matplotlib scikit-learn
```

---

## Usage

```bash
python main.py
```

Part 1 runs first (takes seconds). Part 2 downloads MNIST on the first run (~55 MB, cached afterwards) and fits PCA on 70,000 images, which may take a minute.

---

## Algorithm Details

### K-Means

The algorithm alternates between two steps until convergence:

1. **Assignment** — assign each point to the nearest centroid using squared Euclidean distance (vectorized via broadcasting)
2. **Update** — recompute each centroid as the mean of its assigned points

Initialization: K centroids are sampled uniformly at random from the dataset. Empty clusters are reinitialized randomly to prevent dead centroids.

Convergence is detected when centroid positions stop changing (`np.allclose`). The within-cluster sum of squares (WCSS) is recorded each iteration.

**On convex data** (4 Gaussian blobs), K-Means with k=4 finds the correct clusters in a few iterations.  
**On non-convex data** (two half-moons), K-Means fails — it draws straight-line boundaries and cannot capture the crescent shapes.

### PCA

PCA finds directions of maximum variance in the data:

1. **Center** the data by subtracting the per-feature mean
2. **Compute** the covariance matrix: `C = (X_centered^T · X_centered) / (n-1)`
3. **Eigendecompose** using `np.linalg.eigh` (stable for real symmetric matrices)
4. **Sort** eigenvectors by descending eigenvalue → these are the principal axes V

**Transform:** `X_reduced = X_centered @ V[:, :k]` — projects data into the k-dimensional subspace.

**Reconstruct:** `x_rec = V[:, :k] @ (V[:, :k]^T @ x_centered) + mean` — maps back to the original space.

**Projection matrices:**
- `V^T V ≈ I` — the principal axes are orthonormal (confirmed by the near-identity heatmap)
- `V V^T` — the projection operator in the original space; applying it twice is idempotent

---

## Example Outputs

### Part 1 — K-Means

| Plot | What to expect |
|------|----------------|
| Convex blobs, raw | 4 tight Gaussian clusters clearly separated in 2D |
| K-Means progress (convex) | Centroids converge to cluster centers within 5–10 iterations |
| WCSS curve (convex) | Steep drop then flat — clean convergence |
| Two-moons, raw | Two crescent-shaped clusters that K-Means cannot separate |
| K-Means progress (moons) | Centroid boundary cuts horizontally through both crescents |
| WCSS curve (moons) | Also converges, but at a suboptimal partition |

### Part 2 — PCA on MNIST

| Plot | What to expect |
|------|----------------|
| 2D PCA scatter | 70,000 digits projected to 2D; different digits loosely cluster by region, with some overlap between similar-looking ones (e.g. 3 and 8) |
| V^T V heatmap | Near-identity matrix (bright diagonal, dark off-diagonal) confirming orthonormality |
| V V^T heatmap | Block structure showing the projection subspace footprint in pixel space |
| Reconstruction at 3 dims | Very blurry; only the rough shape of the digit is visible |
| Reconstruction at 10 dims | Blurry but recognizable as the correct digit |
| Reconstruction at 100 dims | Most pixel-level features restored; close to the original |

---

## Suggested Improvements

These are high-value additions that would deepen the project without over-engineering it:

1. **KMeans++ initialization** — seed centroids with distance-proportional probability instead of uniform random sampling. Converges faster, avoids poor initializations, and consistently yields lower final WCSS. One extra private method `_init_centroids_plusplus()`.

2. **Elbow method** — run K-Means for K = 1, 2, ..., 15 and plot final WCSS vs K. The "elbow" in the curve reveals the natural number of clusters without needing ground-truth labels.

3. **Silhouette score** — a per-point metric measuring how well each point fits its assigned cluster compared to the nearest neighboring cluster (+1 = perfect, 0 = boundary, −1 = misclassified). Pairs with the elbow method to confirm results quantitatively.

4. **Explained variance ratio for PCA** — plot `eigenvalues / eigenvalues.sum()` (cumulative) to show what fraction of total data variance is captured by k components. Answers "how many dimensions are enough?" without manual trial-and-error.

5. **DBSCAN comparison on moons** — run `sklearn.cluster.DBSCAN` alongside K-Means on the two-moons dataset and display both results side by side. The contrast perfectly illustrates K-Means' geometric limitation with no extra math required.

6. **Multiple random restarts (`n_init`)** — run K-Means N times with different random seeds and keep the result with the lowest final WCSS. This is what scikit-learn does by default (`n_init=10`) and dramatically stabilizes results on real data.

---

## License

MIT
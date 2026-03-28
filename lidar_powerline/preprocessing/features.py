"""
lidar_powerline.preprocessing.features
========================================
Geometric feature extraction for LiDAR point clouds.

For each point, a local neighbourhood is computed via KD-Tree search and a
3x3 covariance matrix is built from the neighbourhood coordinates. The
eigendecomposition of that matrix yields three principal components and
eigenvalues (λ₀ ≥ λ₁ ≥ λ₂) from which the following features are derived:

    ┌─────────────────────┬──────────────────────────────────────────┐
    │ Feature             │ Formula                                  │
    ├─────────────────────┼──────────────────────────────────────────┤
    │ Linearity           │ (e0 - e1) / e0                           │
    │ Planarity           │ (e1 - e2) / e0                           │
    │ Scattering          │ e2 / e0                                  │
    │ Anisotropy          │ (e0 - e2) / e1                           │
    │ Curvature           │ λ₂ / (λ₀ + λ₁ + λ₂)                     │
    │ Surface variation   │ λ₂ / (λ₀ + λ₁ + λ₂)                     │
    │ Sum of eigenvalues  │ λ₀ + λ₁ + λ₂                             │
    │ Omnivariance        │ (e0 · e1 · e2)^(1/3)                    │
    │ Eigentropy          │ -Σ eᵢ · log(eᵢ)                         │
    └─────────────────────┴──────────────────────────────────────────┘

where eᵢ = λᵢ / (λ₀ + λ₁ + λ₂) are the normalised eigenvalues.
"""

import math

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.neighbors import KDTree


# ---------------------------------------------------------------------------
# KD-Tree neighbourhood + PCA
# ---------------------------------------------------------------------------

def kdtree_point(df: pd.DataFrame, k: int = 250) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute k-nearest-neighbour covariance features for every point.

    Builds a KD-Tree over the XYZ coordinates, queries the k nearest
    neighbours for each point, and computes the PCA of the neighbourhood
    covariance matrix to populate eigenvalue and principal-component columns.

    Args:
        df: DataFrame containing at minimum [x, y, z] columns.
        k:  Number of nearest neighbours to use (default 250).

    Returns:
        Tuple of (enriched DataFrame, index array, distance array).
        The DataFrame gains columns:
            PC1, PC2, PC3        - principal component vectors
            eig_val_0/1/2        - sorted eigenvalues (descending)
            e0, e1, e2           - normalised eigenvalues
            distance             - distance to the k-th neighbour
    """
    X = np.vstack((df["x"].values, df["y"].values, df["z"].values)).T

    tree = KDTree(X, leaf_size=2)
    dist, ind = tree.query(X, k=k)

    PC1_cols, PC2_cols, PC3_cols = [], [], []
    lambda0_cols, lambda1_cols, lambda2_cols = [], [], []
    distance_cols = []

    for i in range(len(df)):
        cov = np.cov(X[ind[i][1:]].T)
        eig_vals, eig_vecs = np.linalg.eig(cov)
        order = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]

        PC1, PC2, PC3 = eig_vecs.T
        PC1_cols.append(PC1)
        PC2_cols.append(PC2)
        PC3_cols.append(PC3)
        lambda0_cols.append(eig_vals[0])
        lambda1_cols.append(eig_vals[1])
        lambda2_cols.append(eig_vals[2])
        distance_cols.append(dist[i][k - 1])

    df = df.copy()
    df["PC1"] = PC1_cols
    df["PC2"] = PC2_cols
    df["PC3"] = PC3_cols
    df["eig_val_0"] = lambda0_cols
    df["eig_val_1"] = lambda1_cols
    df["eig_val_2"] = lambda2_cols
    df["distance"] = distance_cols

    eig_sum = df["eig_val_0"] + df["eig_val_1"] + df["eig_val_2"]
    df["e0"] = df["eig_val_0"] / eig_sum
    df["e1"] = df["eig_val_1"] / eig_sum
    df["e2"] = df["eig_val_2"] / eig_sum

    return df, ind, dist


def construct_local_neighborhood(
    points: np.ndarray, radius: float, k: int
) -> list[np.ndarray]:
    """Build local neighbourhoods around each point using radius search.

    Invalid (NaN/inf) points are filtered before the KD-Tree is constructed.
    If a neighbourhood exceeds ``k`` points, only the first ``k`` are kept.

    Args:
        points: (N, 3) float array of XYZ coordinates.
        radius: Search radius in the same units as the coordinates.
        k:      Maximum neighbourhood size.

    Returns:
        List of (M, 3) arrays, one per valid input point.
    """
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]

    kdtree = KDTree(points)
    neighborhoods = []
    for point in points:
        indices = kdtree.query_radius([point], r=radius)[0]
        if len(indices) > k:
            indices = indices[:k]
        neighborhoods.append(points[indices])

    return neighborhoods


def calculate_eigenvalues(
    local_neighborhood: list[np.ndarray],
) -> tuple[list, list, list, list, list, list]:
    """Compute sorted PCA eigenvalues and eigenvectors for a list of neighbourhoods.

    Args:
        local_neighborhood: Output of ``construct_local_neighborhood``.

    Returns:
        Six lists (PC1, PC2, PC3, λ₀, λ₁, λ₂), one entry per neighbourhood.
    """
    PC1_cols, PC2_cols, PC3_cols = [], [], []
    lambda0_cols, lambda1_cols, lambda2_cols = [], [], []

    for neighborhood in local_neighborhood:
        cov = np.cov(neighborhood.T)
        eig_vals, eig_vecs = np.linalg.eig(cov)
        order = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]

        PC1, PC2, PC3 = eig_vecs.T
        PC1_cols.append(PC1)
        PC2_cols.append(PC2)
        PC3_cols.append(PC3)
        lambda0_cols.append(eig_vals[0])
        lambda1_cols.append(eig_vals[1])
        lambda2_cols.append(eig_vals[2])

    return PC1_cols, PC2_cols, PC3_cols, lambda0_cols, lambda1_cols, lambda2_cols


def kdtree_with_eigenvalues(
    df: pd.DataFrame, radius: float, k: int
) -> pd.DataFrame:
    """Attach eigenvalue-based columns to a DataFrame using radius-based search.

    An alternative to ``kdtree_point`` that uses a fixed search radius rather
    than a fixed neighbour count.

    Args:
        df:     DataFrame with [x, y, z] columns.
        radius: Neighbourhood search radius in metres.
        k:      Maximum number of neighbours per point.

    Returns:
        Enriched DataFrame with PCA and normalised eigenvalue columns.
    """
    points = np.vstack((df["x"].values, df["y"].values, df["z"].values)).T
    neighborhoods = construct_local_neighborhood(points, radius, k)
    PC1, PC2, PC3, l0, l1, l2 = calculate_eigenvalues(neighborhoods)

    df = df.copy()
    df["PC1"] = PC1
    df["PC2"] = PC2
    df["PC3"] = PC3
    df["eig_val_0"] = l0
    df["eig_val_1"] = l1
    df["eig_val_2"] = l2

    eig_sum = df["eig_val_0"] + df["eig_val_1"] + df["eig_val_2"]
    df["e0"] = df["eig_val_0"] / eig_sum
    df["e1"] = df["eig_val_1"] / eig_sum
    df["e2"] = df["eig_val_2"] / eig_sum

    return df


# ---------------------------------------------------------------------------
# High-level feature computation
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all geometric features from normalised eigenvalues.

    Requires that ``kdtree_point`` (or ``kdtree_with_eigenvalues``) has
    already been called so that e0, e1, e2, eig_val_* columns exist.

    Args:
        df: DataFrame with eigenvalue columns populated.

    Returns:
        DataFrame with additional feature columns appended in-place.
    """
    eig_sum = df["eig_val_0"] + df["eig_val_1"] + df["eig_val_2"]

    df["surface_variation"] = df["eig_val_2"] / eig_sum
    df["linearity"] = (df["e0"] - df["e1"]) / df["e0"]
    df["planarity"] = (df["e1"] - df["e2"]) / df["e0"]
    df["scattering"] = df["e2"] / df["e0"]
    df["anisotropy"] = (df["e0"] - df["e2"]) / df["e1"]
    df["curvature"] = df["eig_val_0"] / eig_sum
    df["sum_of_eig"] = eig_sum
    df["omnivariance"] = (df["e0"] * df["e1"] * df["e2"]) ** (1 / 3)
    df["eigentropy"] = df.apply(
        lambda row: (
            -row["e0"] * math.log(row["e0"])
            - row["e1"] * math.log(row["e1"])
            - row["e2"] * math.log(row["e2"])
        ),
        axis=1,
    )
    return df


def compute_density(df: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Estimate local point density within a sphere of given radius.

    Uses Open3D's radius search for efficiency. Density is approximated as:
        density = (0.75 * k) / (π * radius³)

    Args:
        df:     DataFrame with [x, y, z] columns.
        radius: Search radius in metres.

    Returns:
        DataFrame with a new ``density`` column.
    """
    points = np.array(df[["x", "y", "z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    densities = []
    for i in range(df.shape[0]):
        k, _, _ = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        densities.append((0.75 * k) / (3.14159 * radius ** 3))

    df = df.copy()
    df["density"] = densities
    return df


# ---------------------------------------------------------------------------
# Label & DataFrame utilities
# ---------------------------------------------------------------------------

def reassign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remap DALES semantic class IDs to contiguous zero-based integers.

    DALES class mapping:
        2 (ground)      → 0
        3 (vegetation)  → 1
        5 (powerline)   → 2
        6 (fence)       → 3
        7 (pole)        → 4
        8 (building)    → 5

    Args:
        df: DataFrame with a ``sem_class`` column.

    Returns:
        DataFrame with remapped labels, sorted by the new ``sem_class``.
    """
    sem_class_map = {2: 0, 3: 1, 5: 2, 6: 3, 7: 4, 8: 5}

    before = df["sem_class"].value_counts().sort_index()
    print("Class distribution before remapping:\n", before)

    df = df.copy()
    df["sem_class"] = df["sem_class"].replace(sem_class_map)
    df.sort_values(by=["sem_class"], inplace=True)

    after = df["sem_class"].value_counts().sort_index()
    print("Class distribution after remapping:\n", after)

    return df


def create_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model-relevant columns from an enriched DataFrame.

    Returns a clean DataFrame containing only the coordinate, geometric
    feature, intensity, and label columns used by the classifier.

    Args:
        df: Fully enriched DataFrame (after kdtree_point + create_features).

    Returns:
        Subset DataFrame with columns:
            [x, y, z, curvature, linearity, planarity, scattering,
             omnivariance, eigentropy, intensity, sem_class]
    """
    cols = [
        "x", "y", "z",
        "curvature", "linearity", "planarity", "scattering",
        "omnivariance", "eigentropy", "intensity", "sem_class",
    ]
    return df[cols].copy()

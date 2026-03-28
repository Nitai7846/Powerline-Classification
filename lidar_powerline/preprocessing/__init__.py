"""
lidar_powerline.preprocessing
==============================
Subpackage for loading, filtering, and feature-engineering LiDAR point clouds.

Modules:
    io             - Read/write .ply files and convert to pandas DataFrames
    filters        - Ground, building, and vegetation removal
    features       - KD-Tree neighbourhood analysis and geometric feature extraction
    visualization  - 3-D point cloud and neighbourhood visualisation helpers
"""

from .io import ply_to_csv, convert_ply_to_df
from .filters import ground_filtering, building_filter, filter_vegetation, select_vegetation
from .features import (
    kdtree_point,
    create_features,
    compute_density,
    construct_local_neighborhood,
    calculate_eigenvalues,
    kdtree_with_eigenvalues,
    create_dataframe,
    reassign_labels,
)
from .visualization import plot, visualize_local_neighborhood

__all__ = [
    "ply_to_csv",
    "convert_ply_to_df",
    "ground_filtering",
    "building_filter",
    "filter_vegetation",
    "select_vegetation",
    "kdtree_point",
    "create_features",
    "compute_density",
    "construct_local_neighborhood",
    "calculate_eigenvalues",
    "kdtree_with_eigenvalues",
    "create_dataframe",
    "reassign_labels",
    "plot",
    "visualize_local_neighborhood",
]

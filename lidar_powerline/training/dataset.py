"""
lidar_powerline.training.dataset
==================================
Dataset preparation utilities for training the powerline classifier.

The DALES dataset provides labelled aerial LiDAR scenes in PLY format.
Each file contains points from multiple semantic classes. This module
extracts points belonging to a single class across all training scenes,
computes their geometric features, and exports one CSV per class.

DALES semantic class IDs used in this project
----------------------------------------------
    2  - Ground
    5  - Powerline
    7  - Utility pole
    8  - Building

The four merged CSVs are later concatenated and used to train the MLP.

Typical workflow
----------------
::

    # Run once to build per-class CSV files
    python scripts/prepare_dataset.py \\
        --data_dir /path/to/DALESObjects/train \\
        --output_dir /path/to/output \\
        --sem_class 5          # repeat for classes 2, 7, 8
"""

import os

import numpy as np
import pandas as pd

from lidar_powerline.preprocessing.io import convert_ply_to_df
from lidar_powerline.preprocessing.features import kdtree_point, create_features
from lidar_powerline.preprocessing.visualization import plot


# Minimum number of points required to compute KD-Tree features reliably
MIN_POINTS_FOR_FEATURES = 1000


def create_class_dataframe(ply_path: str, sem_class: int) -> pd.DataFrame:
    """Extract and featurise points of a single semantic class from one PLY file.

    Loads the point cloud, filters to the requested class, and (if there are
    enough points) runs KD-Tree neighbourhood analysis + geometric feature
    engineering.

    Args:
        ply_path:  Path to a binary .ply scene file.
        sem_class: DALES semantic class ID to extract.

    Returns:
        Featurised DataFrame for the requested class, or an empty DataFrame
        if the class has fewer than ``MIN_POINTS_FOR_FEATURES`` points.
    """
    df = convert_ply_to_df(ply_path)
    df = df[df["sem_class"] == sem_class]

    if df.shape[0] < MIN_POINTS_FOR_FEATURES:
        return pd.DataFrame()

    df, _, _ = kdtree_point(df)
    df = create_features(df)
    return df


def load_and_merge_csvs(csv_paths: list[str]) -> pd.DataFrame:
    """Concatenate multiple per-class CSV files into a single training DataFrame.

    Args:
        csv_paths: List of paths to CSV files produced by ``prepare_dataset.py``.

    Returns:
        Single concatenated DataFrame with all rows and a reset index.
    """
    dfs = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(dfs, axis=0, ignore_index=True)


def collect_ply_files(directory: str) -> list[str]:
    """Return a sorted list of all .ply file paths in a directory.

    Args:
        directory: Path to a directory containing .ply files.

    Returns:
        Sorted list of absolute .ply file paths.
    """
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(".ply")
    )

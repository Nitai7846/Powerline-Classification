"""
lidar_powerline.detection.tiling
==================================
Spatial tiling utilities for large LiDAR point clouds.

A large aerial scan may contain millions of points. To make Hough-line
detection tractable, the scene is divided into regular square tiles in the
XY plane. Each tile is processed independently, and only tiles that contain
enough Hough-detected line segments are forwarded to the classifier.

Typical usage
-------------
::

    bounding_boxes = tile_generator(ply_path, tile_size=50)
    tile_dfs = filter_dataframes(scene_df, bounding_boxes)
    candidate_tiles = [t for t in tile_dfs if t.shape[0] > 250]
"""

import numpy as np
import pandas as pd
from plyfile import PlyData


BoundingBox = tuple[
    tuple[float, float],  # (x_min, y_min)
    tuple[float, float],  # (x_min, y_max)
    tuple[float, float],  # (x_max, y_min)
    tuple[float, float],  # (x_max, y_max)
]


def tile_generator(ply_path: str, tile_size: float) -> list[BoundingBox]:
    """Partition a PLY scene into a regular grid of square tiles.

    Reads only the vertex extents from the PLY file (does not load all data
    into memory) and returns a list of axis-aligned bounding boxes.

    Args:
        ply_path:  Path to the binary .ply file.
        tile_size: Side length of each tile in metres.

    Returns:
        List of bounding boxes, each represented as:
            ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
    """
    with open(ply_path, "rb") as f:
        plydata = PlyData.read(f)

    vertices = plydata.elements[0]

    x_min = np.int32(vertices["x"].min())
    x_max = np.int32(vertices["x"].max())
    num_tiles_x = (x_max - x_min) // tile_size + 1
    x_coords = np.linspace(x_min, x_min + tile_size * num_tiles_x, num=num_tiles_x + 1)

    y_min = np.int32(vertices["y"].min())
    y_max = np.int32(vertices["y"].max())
    num_tiles_y = (y_max - y_min) // tile_size + 1
    y_coords = np.linspace(y_min, y_min + tile_size * num_tiles_y, num=num_tiles_y + 1)

    bounding_boxes: list[BoundingBox] = []
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            bbox: BoundingBox = (
                (x_coords[i],     y_coords[j]),
                (x_coords[i],     y_coords[j + 1]),
                (x_coords[i + 1], y_coords[j]),
                (x_coords[i + 1], y_coords[j + 1]),
            )
            bounding_boxes.append(bbox)

    return bounding_boxes


def generate_3d_grid(
    df: pd.DataFrame,
    bounding_boxes: list[BoundingBox],
    grid_number: int,
) -> pd.DataFrame:
    """Extract points from a single tile by iterating over rows.

    .. note::
        For large DataFrames prefer ``filter_dataframes`` which uses
        vectorised boolean indexing.

    Args:
        df:            Full scene DataFrame with [x, y, z, ...] columns.
        bounding_boxes: Output of ``tile_generator``.
        grid_number:   Index of the desired tile.

    Returns:
        DataFrame containing only the points within the selected tile.
    """
    (x_min, y_min), (_, y_max), (x_max, _), _ = bounding_boxes[grid_number]
    selected = [
        row for _, row in df.iterrows()
        if x_min <= row["x"] <= x_max and y_min <= row["y"] <= y_max
    ]
    new_df = pd.DataFrame.from_records(selected)
    new_df.columns = df.columns
    return new_df


def filter_dataframes(
    df: pd.DataFrame,
    bounding_boxes: list[BoundingBox],
) -> list[pd.DataFrame]:
    """Split a DataFrame into per-tile sub-DataFrames (vectorised).

    Args:
        df:            Full scene DataFrame with [x, y, z, ...] columns.
        bounding_boxes: Output of ``tile_generator``.

    Returns:
        List of DataFrames, one per bounding box. Empty tiles produce empty
        DataFrames.
    """
    result = []
    for bbox in bounding_boxes:
        (x_min, y_min), (_, y_max), (x_max, _), _ = bbox
        mask = (
            (df["x"] >= x_min) & (df["x"] <= x_max) &
            (df["y"] >= y_min) & (df["y"] <= y_max)
        )
        result.append(df[mask])
    return result

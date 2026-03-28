"""
lidar_powerline.preprocessing.io
=================================
Input/output utilities for LiDAR point cloud data.

Supports reading binary PLY files (DALES dataset format) and converting them
to pandas DataFrames or CSV files for downstream processing.

Expected PLY vertex fields:
    x, y, z         - 3-D coordinates (float)
    intensity        - Return intensity (float)
    sem_class        - Semantic class label (int)
    ins_class        - Instance class label (int)
"""

import csv

import numpy as np
import pandas as pd
from plyfile import PlyData


def ply_to_csv(ply_path: str, csv_path: str) -> None:
    """Convert a binary PLY file to a CSV file.

    Reads vertex data (x, y, z, intensity, sem_class, ins_class) from a PLY
    file and writes it row-by-row to a CSV file.

    Args:
        ply_path: Path to the input .ply file.
        csv_path: Path to the output .csv file (will be created or overwritten).
    """
    with open(ply_path, "rb") as f:
        plydata = PlyData.read(f)

    vertices = plydata.elements[0]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "z", "intensity", "sem_class", "ins_class"])
        for vertex in vertices:
            writer.writerow([
                vertex[0], vertex[1], vertex[2],
                vertex[3], vertex[4], vertex[5],
            ])


def convert_ply_to_df(ply_path: str) -> pd.DataFrame:
    """Load a PLY point cloud into a pandas DataFrame.

    Args:
        ply_path: Path to the binary .ply file.

    Returns:
        DataFrame with columns [x, y, z, intensity, sem_class, ins_class].
    """
    with open(ply_path, "rb") as f:
        plydata = PlyData.read(f)

    vertices = plydata.elements[0]

    data = {
        "x": vertices["x"],
        "y": vertices["y"],
        "z": vertices["z"],
        "intensity": vertices["intensity"],
        "sem_class": vertices["sem_class"],
        "ins_class": vertices["ins_class"],
    }

    return pd.DataFrame(data)

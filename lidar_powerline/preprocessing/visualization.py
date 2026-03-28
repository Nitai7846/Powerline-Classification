"""
lidar_powerline.preprocessing.visualization
=============================================
Visualisation helpers for LiDAR point clouds and local neighbourhoods.

Wraps Open3D (interactive 3-D viewer) and Plotly (interactive scatter3d)
to provide quick visual inspection at any stage of the pipeline.
"""

import numpy as np
import open3d as o3d
import pandas as pd
import plotly.graph_objects as go


def plot(df: pd.DataFrame) -> None:
    """Display a point cloud interactively using the Open3D viewer.

    Opens a native window showing all points in the DataFrame as a
    white point cloud. Close the window to continue execution.

    Args:
        df: DataFrame with at minimum [x, y, z] columns.
    """
    points = np.array(df[["x", "y", "z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


def visualize_local_neighborhood(
    local_neighborhood: list[np.ndarray],
) -> None:
    """Render local point neighbourhoods as an interactive Plotly scatter plot.

    Each neighbourhood is drawn as a separate scatter3d trace, enabling
    colour-coded inspection of multiple neighbourhoods simultaneously.

    Args:
        local_neighborhood: List of (M, 3) arrays as returned by
            ``construct_local_neighborhood``.
    """
    fig = go.Figure()

    for neighborhood in local_neighborhood:
        fig.add_trace(
            go.Scatter3d(
                x=neighborhood[:, 0],
                y=neighborhood[:, 1],
                z=neighborhood[:, 2],
                mode="markers",
                marker=dict(size=2),
            )
        )

    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()

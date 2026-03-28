"""
lidar_powerline.preprocessing.filters
=======================================
Spatial filters for removing non-powerline point classes from a LiDAR scene.

Three complementary filters are applied sequentially before powerline detection:

1. **Ground filter** (CSF - Cloth Simulation Filter):
   Separates ground returns from above-ground objects using a physics-based
   cloth draping simulation.

2. **Building filter** (planar patch detection):
   Detects large planar surfaces (rooftops, walls) via normal estimation and
   oriented bounding box fitting, then removes those points.

3. **Vegetation filter** (2-D density grid):
   Projects points onto a horizontal grid and removes cells whose neighbourhood
   density exceeds a threshold, effectively isolating dense canopy clusters.
"""

import numpy as np
import open3d as o3d
import pandas as pd
import CSF


def ground_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ground returns using the Cloth Simulation Filter (CSF).

    Uses the CSF algorithm to separate ground and non-ground points.
    Only non-ground points are returned for further processing.

    Args:
        df: DataFrame with columns [x, y, z, intensity, sem_class].

    Returns:
        DataFrame containing only non-ground points, with the same columns.
    """
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = 0.5

    print(f"Ground filter — input points: {len(df):,}")

    points = np.array(df[["x", "y", "z", "intensity", "sem_class"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    csf.setPointCloud(pcd.points)
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    df_filtered = pd.DataFrame(points[non_ground])
    df_filtered.columns = ["x", "y", "z", "intensity", "sem_class"]

    print(f"Ground filter — output points: {len(df_filtered):,}")
    return df_filtered


def building_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove building points via planar patch detection.

    Estimates surface normals and detects oriented planar patches (rooftops,
    walls). Points that fall within any detected patch bounding box are
    removed. The result is merged back with the original attribute columns.

    Args:
        df: DataFrame with columns [x, y, z, intensity, sem_class, ins_class]
            (or a subset thereof).

    Returns:
        DataFrame with building points removed.
    """
    points = np.array(df[["x", "y", "z"]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0,
        min_num_points=10,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
    )
    print(f"Building filter — detected {len(oboxes)} planar patches")

    pcd_filtered = pcd
    for obox in oboxes:
        indices = obox.get_point_indices_within_bounding_box(pcd_filtered.points)
        pcd_filtered = pcd_filtered.select_by_index(indices, invert=True)

    filtered_array = np.array(pcd_filtered.points)
    df_new = pd.DataFrame(filtered_array, columns=["x", "y", "z"])

    attribute_cols = [c for c in ["x", "y", "z", "sem_class", "ins_class", "intensity"] if c in df.columns]
    merged_df = pd.merge(df_new, df[attribute_cols], on=["x", "y", "z"], how="left")
    return merged_df


def filter_vegetation(
    df: pd.DataFrame,
    cell_size: float = 1.0,
    density_threshold: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Remove dense vegetation clusters using a 2-D horizontal density grid.

    Projects the point cloud onto the XY plane, bins points into grid cells,
    and removes cells whose 3x3 neighbourhood contains more than
    ``density_threshold`` high-density cells (indicative of canopy).

    Args:
        df: DataFrame with columns [x, y, z, sem_class, intensity].
        cell_size: Side length of each grid cell in metres.
        density_threshold: Minimum point count per cell to be considered dense.

    Returns:
        Tuple of:
            - Filtered DataFrame (dense vegetation removed).
            - 2-D numpy density grid (for optional visualisation).
    """
    return _apply_vegetation_grid(df, cell_size, density_threshold, remove_dense=True)


def select_vegetation(
    df: pd.DataFrame,
    cell_size: float = 1.0,
    density_threshold: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Isolate dense vegetation clusters (inverse of ``filter_vegetation``).

    Keeps only points belonging to cells surrounded by other high-density
    cells, effectively selecting canopy regions.

    Args:
        df: DataFrame with columns [x, y, z, sem_class, intensity].
        cell_size: Side length of each grid cell in metres.
        density_threshold: Minimum point count per cell to be considered dense.

    Returns:
        Tuple of:
            - DataFrame containing only vegetation points.
            - 2-D numpy density grid.
    """
    return _apply_vegetation_grid(df, cell_size, density_threshold, remove_dense=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_vegetation_grid(
    df: pd.DataFrame,
    cell_size: float,
    density_threshold: int,
    remove_dense: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Shared grid logic for both filter_vegetation and select_vegetation."""
    points = np.array(df[["x", "y", "z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    point_cloud = np.array(pcd.points)

    grid_size = int(np.ceil(np.max(point_cloud[:, :2]) / cell_size))
    grid = np.zeros((grid_size, grid_size), dtype=int)

    for point in point_cloud:
        x, y, _ = point
        grid[int(x / cell_size), int(y / cell_size)] += 1

    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] > density_threshold:
                neighbor_count = np.sum(
                    grid[
                        max(i - 1, 0):min(i + 2, grid_size),
                        max(j - 1, 0):min(j + 2, grid_size),
                    ] > density_threshold
                )
                if remove_dense and neighbor_count > 3:
                    grid[i, j] = 0
                elif not remove_dense and neighbor_count < 3:
                    grid[i, j] = 0

    mask = np.zeros(len(point_cloud), dtype=bool)
    for idx, point in enumerate(point_cloud):
        x, y, _ = point
        if grid[int(x / cell_size), int(y / cell_size)] > 0:
            mask[idx] = True

    filtered_cloud = point_cloud[mask]

    df_new = pd.DataFrame(filtered_cloud, columns=["x", "y", "z"])
    attribute_cols = [c for c in ["x", "y", "z", "sem_class", "intensity"] if c in df.columns]
    merged_df = pd.merge(df_new, df[attribute_cols], on=["x", "y", "z"], how="left")
    return merged_df, grid

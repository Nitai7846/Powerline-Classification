"""
lidar_powerline.detection
==========================
Subpackage for powerline candidate detection.

After the preprocessing filters have removed ground, buildings, and dense
vegetation, this subpackage tiles the remaining point cloud and applies a
Hough-line transform to identify spatial regions likely to contain powerlines.

Modules:
    tiling  - Spatial tiling of large point clouds into manageable grid cells
    hough   - Voxel projection and probabilistic Hough-line detection
"""

from .tiling import tile_generator, generate_3d_grid, filter_dataframes
from .hough import voxel2image, hough_transform

__all__ = [
    "tile_generator",
    "generate_3d_grid",
    "filter_dataframes",
    "voxel2image",
    "hough_transform",
]

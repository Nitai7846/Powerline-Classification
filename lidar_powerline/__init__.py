"""
lidar_powerline
===============
A Python package for powerline detection in aerial LiDAR point clouds.

Pipeline:
    1. Load .ply point cloud data
    2. Filter ground, buildings, and vegetation
    3. Extract geometric features via KD-Tree neighbourhood analysis
    4. Tile the scene and apply Hough-line detection to candidate regions
    5. Classify remaining points with a trained MLP (vegetation / powerline / pole / building)

Subpackages:
    preprocessing  - I/O, spatial filters, feature engineering, visualisation
    detection      - Tiling, voxelisation, and Hough-line detection
    training       - Dataset preparation, model definition, and training utilities
"""

__version__ = "1.0.0"
__author__ = "Nitai Shah"

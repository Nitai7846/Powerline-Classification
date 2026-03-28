"""
lidar_powerline.training
=========================
Subpackage for dataset preparation and model training.

Modules:
    dataset  - Load, merge, and split per-class PLY data into training CSVs
    model    - MLP architecture, training helpers, and inference utilities
"""

from .dataset import create_class_dataframe, load_and_merge_csvs
from .model import build_model, select_features, apply_encoding, apply_scaling

__all__ = [
    "create_class_dataframe",
    "load_and_merge_csvs",
    "build_model",
    "select_features",
    "apply_encoding",
    "apply_scaling",
]

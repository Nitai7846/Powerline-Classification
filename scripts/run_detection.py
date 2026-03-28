"""
run_detection.py
================
End-to-end powerline detection pipeline for a single LiDAR scene.

Pipeline
--------
1. Load the .ply scene.
2. Remove ground (CSF), vegetation (density grid), and buildings (planar patches).
3. Tile the filtered scene into 50 m × 50 m cells.
4. Voxelise each tile and apply the Hough line transform.
5. Forward tiles with ≥ ``--line_threshold`` detected lines to the MLP.
6. Run the MLP classifier; collect powerline-labelled points.
7. Visualise the final powerline point cloud.

Usage
-----
    python scripts/run_detection.py \\
        --ply_file    /path/to/scene.ply \\
        --model_path  weights/model.h5 \\
        --scaler_path weights/scaler_file.pkl \\
        --tile_size   50 \\
        --voxel_size  0.2 \\
        --line_threshold 4
"""

import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from lidar_powerline.preprocessing.filters import (
    filter_vegetation,
    ground_filtering,
)
from lidar_powerline.preprocessing.features import create_features, kdtree_point
from lidar_powerline.preprocessing.visualization import plot
from lidar_powerline.detection.tiling import filter_dataframes, tile_generator
from lidar_powerline.detection.hough import hough_transform, voxel2image
from lidar_powerline.training.model import FEATURE_LIST, load_scaler
from lidar_powerline.preprocessing.io import convert_ply_to_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect powerlines in a LiDAR point cloud."
    )
    parser.add_argument("--ply_file",       required=True, help="Path to input .ply scene.")
    parser.add_argument("--model_path",     required=True, help="Path to trained model.h5.")
    parser.add_argument("--scaler_path",    required=True, help="Path to fitted scaler_file.pkl.")
    parser.add_argument("--tile_size",      type=float, default=50.0)
    parser.add_argument("--voxel_size",     type=float, default=0.2)
    parser.add_argument("--line_threshold", type=int,   default=4,
                        help="Minimum Hough lines to classify a tile.")
    parser.add_argument("--min_tile_points", type=int,  default=250,
                        help="Minimum points in a tile to run the classifier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load scene                                                        #
    # ------------------------------------------------------------------ #
    print(f"Loading: {args.ply_file}")
    df = convert_ply_to_df(args.ply_file)
    print(f"Total points: {len(df):,}")
    plot(df)

    # ------------------------------------------------------------------ #
    # 2. Sequential filtering                                              #
    # ------------------------------------------------------------------ #
    df_no_ground = ground_filtering(df)
    df_filtered, _ = filter_vegetation(df_no_ground)
    print(f"Points after filtering: {len(df_filtered):,}")
    plot(df_filtered)

    # ------------------------------------------------------------------ #
    # 3. Tile the filtered scene                                           #
    # ------------------------------------------------------------------ #
    bounding_boxes = tile_generator(args.ply_file, args.tile_size)
    tile_dfs = filter_dataframes(df_no_ground, bounding_boxes)
    print(f"Tiles generated: {len(tile_dfs)}")

    # ------------------------------------------------------------------ #
    # 4. Hough-line candidate selection                                    #
    # ------------------------------------------------------------------ #
    line_counts = []
    for tile_df in tile_dfs:
        image, _, _ = voxel2image(tile_df, args.voxel_size)
        if image is not None:
            n_lines, _ = hough_transform(image)
            line_counts.append(n_lines)
        else:
            line_counts.append(0)

    candidate_indices = [i for i, n in enumerate(line_counts) if n > args.line_threshold]
    candidate_tiles = [tile_dfs[i] for i in candidate_indices]
    print(f"Candidate tiles (>{args.line_threshold} lines): {len(candidate_tiles)}")

    # ------------------------------------------------------------------ #
    # 5. MLP classification                                                #
    # ------------------------------------------------------------------ #
    scaler = load_scaler(args.scaler_path)
    model = keras.models.load_model(args.model_path)

    powerline_dfs = []
    for tile_df in candidate_tiles:
        if tile_df.shape[0] < args.min_tile_points:
            continue

        tile_df, _, _ = kdtree_point(tile_df)
        tile_df = create_features(tile_df)

        X = scaler.transform(tile_df[FEATURE_LIST])
        predictions = model.predict(X)
        labels = tf.argmax(predictions, axis=1).numpy()
        tile_df["predicted_class"] = labels

        powerlines = tile_df[tile_df["predicted_class"] == 1]
        powerline_dfs.append(powerlines)

    # ------------------------------------------------------------------ #
    # 6. Visualise results                                                 #
    # ------------------------------------------------------------------ #
    if powerline_dfs:
        powerline_final = pd.concat(powerline_dfs, ignore_index=True)
        print(f"\nDetected powerline points: {len(powerline_final):,}")
        plot(powerline_final)
    else:
        print("No powerline points detected.")


if __name__ == "__main__":
    main()

"""
train_model.py
==============
Train the MLP powerline classifier on the merged per-class CSV files
produced by ``prepare_dataset.py``.

The script:
    1. Loads four per-class CSVs (vegetation, powerline, pole, building).
    2. Remaps labels to contiguous integers (0–3).
    3. Drops NaN rows.
    4. Splits into 80 / 20 train-test sets.
    5. Fits and saves a MinMaxScaler.
    6. Trains the MLP for the specified number of epochs.
    7. Evaluates and prints a classification report.
    8. Saves the trained model as ``model.h5``.

Usage
-----
    python scripts/train_model.py \\
        --vegetation_csv  /path/to/Vegetation_Merged.csv \\
        --powerline_csv   /path/to/Powerline_Merged.csv \\
        --pole_csv        /path/to/Pole_Merged.csv \\
        --building_csv    /path/to/Building_Merged.csv \\
        --output_dir      ./weights \\
        --epochs          10
"""

import argparse
import os

import pandas as pd

from lidar_powerline.training.dataset import load_and_merge_csvs
from lidar_powerline.training.model import (
    FEATURE_LIST,
    TARGET_COL,
    apply_encoding,
    apply_scaling,
    build_model,
    evaluate_model,
    select_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LiDAR powerline MLP classifier."
    )
    parser.add_argument("--vegetation_csv", required=True)
    parser.add_argument("--powerline_csv",  required=True)
    parser.add_argument("--pole_csv",       required=True)
    parser.add_argument("--building_csv",   required=True)
    parser.add_argument(
        "--output_dir", default="weights",
        help="Directory to save model.h5 and scaler_file.pkl.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load and merge class CSVs                                         #
    # ------------------------------------------------------------------ #
    csv_paths = [
        args.vegetation_csv,
        args.powerline_csv,
        args.pole_csv,
        args.building_csv,
    ]
    merged_df = load_and_merge_csvs(csv_paths)
    print(f"Merged dataset — {len(merged_df):,} points")

    # ------------------------------------------------------------------ #
    # 2. Remap labels (DALES IDs → 0-based) and drop NaN rows             #
    # ------------------------------------------------------------------ #
    sem_class_map = {2: 0, 5: 1, 7: 2, 8: 3}
    merged_df[TARGET_COL] = merged_df[TARGET_COL].replace(sem_class_map)
    merged_df = merged_df.dropna()
    print(f"After cleaning — {len(merged_df):,} points")
    print("Class distribution:\n", merged_df[TARGET_COL].value_counts().sort_index())

    # ------------------------------------------------------------------ #
    # 3. Train / test split                                                #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = select_features(merged_df)
    vis_df = merged_df.loc[X_test.index].copy()

    y_train_enc, y_test_enc = apply_encoding(y_train, y_test)

    scaler_path = os.path.join(args.output_dir, "scaler_file.pkl")
    X_train_scaled, X_test_scaled = apply_scaling(X_train, X_test, scaler_path)
    print(f"Scaler saved → {scaler_path}")

    # ------------------------------------------------------------------ #
    # 4. Build, train, and save the model                                  #
    # ------------------------------------------------------------------ #
    model = build_model(num_features=len(FEATURE_LIST))
    model.summary()

    history = model.fit(X_train_scaled, y_train_enc, epochs=args.epochs, verbose=1)

    model_path = os.path.join(args.output_dir, "model.h5")
    model.save(model_path)
    print(f"Model saved → {model_path}")

    # ------------------------------------------------------------------ #
    # 5. Evaluate                                                          #
    # ------------------------------------------------------------------ #
    evaluate_model(model, X_test_scaled, y_test_enc, vis_df)


if __name__ == "__main__":
    main()

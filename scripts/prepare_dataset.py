"""
prepare_dataset.py
==================
Extract and featurise points of a given semantic class from all PLY training
files and export the result as a single merged CSV.

Run this script once per class before training:

    python scripts/prepare_dataset.py \\
        --data_dir /path/to/DALESObjects/train \\
        --output_dir /path/to/output \\
        --sem_class 5 \\
        --output_name Powerline_Merged.csv

DALES class IDs
---------------
    2  → Ground
    5  → Powerline
    7  → Utility pole
    8  → Building

Usage
-----
    python scripts/prepare_dataset.py --help
"""

import argparse
import os

import pandas as pd

from lidar_powerline.training.dataset import collect_ply_files, create_class_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-class feature CSV from DALES PLY training files."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing .ply training scene files.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where the merged CSV will be saved.",
    )
    parser.add_argument(
        "--sem_class", type=int, required=True,
        help="DALES semantic class ID to extract (e.g. 5 for powerline).",
    )
    parser.add_argument(
        "--output_name", default=None,
        help="Output CSV filename. Defaults to 'Class<sem_class>_Merged.csv'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ply_files = collect_ply_files(args.data_dir)
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in: {args.data_dir}")

    print(f"Found {len(ply_files)} PLY files. Extracting class {args.sem_class}...")

    result_dfs = []
    for i, ply_path in enumerate(ply_files):
        print(f"  [{i + 1}/{len(ply_files)}] {os.path.basename(ply_path)}")
        df = create_class_dataframe(ply_path, args.sem_class)
        if not df.empty:
            result_dfs.append(df)

    if not result_dfs:
        print("No qualifying points found. Exiting.")
        return

    combined_df = pd.concat(result_dfs, ignore_index=True)
    print(f"\nTotal points extracted: {len(combined_df):,}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name or f"Class{args.sem_class}_Merged.csv"
    output_path = os.path.join(args.output_dir, output_name)
    combined_df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()

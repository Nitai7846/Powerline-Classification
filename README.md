[README.md](https://github.com/user-attachments/files/26320664/README.md)
# LiDAR Powerline Detection

A Python pipeline for detecting **powerlines**, **poles**, **buildings**, and **vegetation** in aerial LiDAR point clouds using geometric feature engineering and a multi-layer perceptron (MLP) classifier.

Trained and evaluated on the [**DALES dataset**](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php) — a large-scale aerial LiDAR dataset with point-level semantic labels.

<img width="901" height="843" alt="Screenshot 2023-05-20 at 10 11 39 AM" src="https://github.com/user-attachments/assets/27047b3c-e712-4f9a-9168-b545b318b543" />





---

## Pipeline Overview

```
Raw .ply Scene
      │
      ▼
┌─────────────────────┐
│  Ground Filtering   │  Cloth Simulation Filter (CSF)
│  (CSF Algorithm)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Building Filtering  │  Planar patch detection (Open3D)
│ (Planar Patches)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│Vegetation Filtering │  2-D horizontal density grid
│ (Density Grid)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Spatial Tiling     │  50 m × 50 m grid cells
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Hough Line Filter  │  Voxelise tile → top-down image → HoughLinesP
│  (Candidate Select) │  Tiles with ≥ N lines → candidate set
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ KD-Tree Feature Eng │  12 geometric features per point
│ (Eigenvalue-based)  │  (linearity, planarity, curvature, eigentropy…)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   MLP Classifier    │  4-class softmax (vegetation / powerline /
│   (TensorFlow/Keras)│   pole / building)
└─────────┬───────────┘
          │
          ▼
   Powerline Point Cloud
```

---

## Geometric Features

For each point, the 250 nearest neighbours are found via KD-Tree. A 3×3 covariance matrix is built from these neighbours and decomposed into eigenvalues (λ₀ ≥ λ₁ ≥ λ₂) and eigenvectors. The following features are computed:

| Feature            | Formula                              |
|--------------------|--------------------------------------|
| Linearity          | (e₀ − e₁) / e₀                       |
| Planarity          | (e₁ − e₂) / e₀                       |
| Scattering         | e₂ / e₀                              |
| Anisotropy         | (e₀ − e₂) / e₁                       |
| Curvature          | λ₀ / Σλ                              |
| Surface Variation  | λ₂ / Σλ                              |
| Sum of Eigenvalues | λ₀ + λ₁ + λ₂                         |
| Omnivariance       | (e₀ · e₁ · e₂)^(1/3)               |
| Eigentropy         | −Σ eᵢ · log(eᵢ)                     |

where eᵢ = λᵢ / (λ₀ + λ₁ + λ₂).

Together with `z`, `intensity`, and `distance` (to the 250th neighbour), this gives **12 input features** to the MLP.

---

## Model Architecture

```
Input (12) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(32, ReLU)
           → Dense(16, ReLU) → Dense(4, Softmax)

Optimiser : Adam
Loss      : Categorical cross-entropy
Classes   : 0=Vegetation  1=Powerline  2=Pole  3=Building
```

---

## Repository Structure

```
lidar-powerline-detection/
├── lidar_powerline/              # Main Python package
│   ├── preprocessing/
│   │   ├── io.py                 # PLY → DataFrame / CSV
│   │   ├── filters.py            # Ground, building, vegetation filters
│   │   ├── features.py           # KD-Tree & geometric feature engineering
│   │   └── visualization.py      # Open3D / Plotly visualisation helpers
│   ├── detection/
│   │   ├── tiling.py             # Spatial tiling of the scene
│   │   └── hough.py              # Voxel projection + Hough line detection
│   └── training/
│       ├── dataset.py            # Per-class CSV preparation
│       └── model.py              # MLP definition, training & eval utilities
├── scripts/
│   ├── prepare_dataset.py        # Build per-class feature CSVs
│   ├── train_model.py            # Train and save the MLP
│   └── run_detection.py          # Full inference pipeline
├── configs/
│   └── default_config.yaml       # All hyperparameters in one place
├── requirements.txt
└── README.md
```

---

## Dataset

This project was developed using the **DALES** (Dayton Annotated LiDAR Earth Scan) dataset.

- 40 aerial LiDAR tiles covering 10 km²
- ~505 million labelled points
- 8 semantic classes: ground, vegetation, cars, trucks, powerlines, fences, poles, buildings

Download: [DALES Dataset](https://udayton.edu/engineering/research/centers/vision_lab/research/was_data_analysis_and_processing/dale.php)

**Semantic class IDs used in this project:**

| ID | Class      | Remapped |
|----|------------|----------|
| 2  | Ground     | —        |
| 5  | Powerline  | 1        |
| 7  | Pole       | 2        |
| 8  | Building   | 3        |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/lidar-powerline-detection.git
cd lidar-powerline-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** Open3D requires Python 3.8–3.11. TensorFlow GPU support requires CUDA 11.2+.

---

## Usage

### Step 1 — Prepare per-class CSVs

Run once per semantic class on the DALES training split:

```bash
python scripts/prepare_dataset.py \
    --data_dir /path/to/DALESObjects/train \
    --output_dir data/merged \
    --sem_class 5 \
    --output_name Powerline_Merged.csv

# Repeat for classes 2 (vegetation), 7 (pole), 8 (building)
```

### Step 2 — Train the model

```bash
python scripts/train_model.py \
    --vegetation_csv data/merged/Vegetation_Merged.csv \
    --powerline_csv  data/merged/Powerline_Merged.csv \
    --pole_csv       data/merged/Pole_Merged.csv \
    --building_csv   data/merged/Building_Merged.csv \
    --output_dir     weights \
    --epochs         10
```

Outputs: `weights/model.h5` and `weights/scaler_file.pkl`

### Step 3 — Run detection on a new scene

```bash
python scripts/run_detection.py \
    --ply_file    /path/to/scene.ply \
    --model_path  weights/model.h5 \
    --scaler_path weights/scaler_file.pkl \
    --tile_size   50 \
    --line_threshold 4
```

---

## Configuration

All tunable parameters live in `configs/default_config.yaml`. Key settings:

| Parameter                        | Default | Description                                   |
|----------------------------------|---------|-----------------------------------------------|
| `filters.ground.cloth_resolution`| 0.5 m   | CSF cloth resolution                          |
| `filters.vegetation.cell_size`   | 1.0 m   | Density grid cell size                        |
| `detection.tile_size`            | 50 m    | Tile side length                              |
| `detection.voxel_size`           | 0.2 m   | Voxel size for top-down projection            |
| `detection.line_threshold`       | 4       | Min Hough lines to flag a tile as candidate   |
| `training.epochs`                | 10      | MLP training epochs                           |
| `features.kdtree.k_neighbours`   | 250     | KD-Tree neighbourhood size                    |

---

## Dependencies

| Library        | Purpose                              |
|----------------|--------------------------------------|
| Open3D         | Point cloud I/O and filtering        |
| plyfile        | Binary PLY reading                   |
| CSF            | Cloth Simulation ground filter       |
| scikit-learn   | KD-Tree, MinMaxScaler, metrics       |
| TensorFlow     | MLP training and inference           |
| OpenCV         | Gaussian blur and Hough transform    |
| scikit-image   | Edge detection utilities             |
| Plotly         | Interactive 3-D neighbourhood viewer |
| pandas / numpy | Data manipulation                    |

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

- **DALES Dataset** — University of Dayton Vision Lab
- **CSF Algorithm** — Zhang et al., "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation" (2016)
- **Open3D** — Zhou et al., "Open3D: A Modern Library for 3D Data Processing" (2018)

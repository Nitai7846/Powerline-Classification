"""
lidar_powerline.training.model
================================
MLP model definition and training/inference utilities.

Architecture
------------
The classifier is a fully-connected feed-forward network trained on 12
geometric features extracted from the KD-Tree neighbourhood of each point:

    Input (12) → Dense(64, ReLU) → Dense(64, ReLU) → Dense(32, ReLU)
              → Dense(16, ReLU) → Dense(4, Softmax)

Output classes (after label remapping):
    0 - Vegetation
    1 - Powerline
    2 - Pole
    3 - Building

The model is trained with categorical cross-entropy and the Adam optimiser.
A MinMaxScaler fitted on the training split is saved alongside the model
weights so that inference uses the same feature scale.
"""

import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


# Feature columns used by the model (must match those used during training)
FEATURE_LIST = [
    "z", "intensity", "distance",
    "surface_variation", "linearity", "planarity",
    "scattering", "anisotropy", "curvature",
    "sum_of_eig", "omnivariance", "eigentropy",
]

TARGET_COL = "sem_class"

# Class index → human-readable label
CLASS_LABELS = {0: "vegetation", 1: "powerline", 2: "pole", 3: "building"}


def build_model(num_features: int = 12, num_classes: int = 4) -> keras.Model:
    """Construct and compile the MLP classifier.

    Args:
        num_features: Number of input features (default 12).
        num_classes:  Number of output classes (default 4).

    Returns:
        Compiled Keras Sequential model.
    """
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(num_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def select_features(
    df: pd.DataFrame,
    feature_list: list[str] = FEATURE_LIST,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train/test feature and label arrays.

    Args:
        df:           Fully featurised DataFrame.
        feature_list: Columns to use as model input.
        target_col:   Column to use as the label.
        test_size:    Fraction of data held out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X = df[feature_list]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def apply_encoding(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encode label arrays for categorical cross-entropy.

    Args:
        y_train: Integer class labels for the training set.
        y_test:  Integer class labels for the test set.

    Returns:
        (y_train_onehot, y_test_onehot)
    """
    return np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)


def apply_scaling(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler_path: str = "scaler_file.pkl",
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a MinMaxScaler on the training set and apply it to both splits.

    The fitted scaler is serialised to ``scaler_path`` so it can be reloaded
    for inference without recomputing feature statistics.

    Args:
        X_train:     Training feature matrix.
        X_test:      Test feature matrix.
        scaler_path: Destination path for the pickled scaler.

    Returns:
        (X_train_scaled, X_test_scaled)
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled


def load_scaler(scaler_path: str) -> MinMaxScaler:
    """Load a previously fitted MinMaxScaler from disk.

    Args:
        scaler_path: Path to a pickled MinMaxScaler.

    Returns:
        Loaded MinMaxScaler instance.
    """
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test_onehot: np.ndarray,
    vis_df: pd.DataFrame | None = None,
) -> None:
    """Print a classification report and confusion matrix for the test set.

    Args:
        model:        Trained Keras model.
        X_test:       Scaled test features.
        y_test_onehot: One-hot encoded test labels.
        vis_df:       Optional DataFrame aligned with X_test for spatial
                      analysis (must have a ``sem_class`` column).
    """
    y_pred_prob = model.predict(X_test)
    y_pred = tf.argmax(y_pred_prob, axis=1).numpy()
    y_true = tf.argmax(y_test_onehot, axis=1).numpy()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(CLASS_LABELS.values())))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    if vis_df is not None:
        vis_df = vis_df.copy()
        vis_df["predicted"] = y_pred
        print("\nSpatial check — predicted class distribution:")
        print(vis_df["predicted"].value_counts())

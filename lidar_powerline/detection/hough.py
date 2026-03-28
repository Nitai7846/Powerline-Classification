"""
lidar_powerline.detection.hough
=================================
Hough-line-based powerline candidate detection.

The approach converts a 3-D tile of filtered points into a 2-D top-down
binary image via voxelisation, then applies a probabilistic Hough line
transform to detect linear structures (powerlines viewed from above).

Pipeline per tile
-----------------
1. ``voxel2image``    — voxelise the tile and project onto the XY plane.
2. ``hough_transform`` — Gaussian blur → Canny edges → HoughLinesP.
3. Tiles with more than ``line_threshold`` detected lines are forwarded
   to the MLP classifier as powerline candidates.

The Hough transform is sensitive to the voxel size and the line detection
parameters. Default values have been tuned on the DALES dataset.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd


def voxel2image(
    df: pd.DataFrame,
    voxel_size: float,
) -> tuple[np.ndarray | None, object, list]:
    """Project a voxelised point cloud tile onto a 2-D top-down binary image.

    Converts the tile to an Open3D VoxelGrid, extracts the (x, y) grid
    indices of each occupied voxel, and paints those pixels white (255) on
    a square black image.

    Args:
        df:         DataFrame with [x, y, z] columns representing one tile.
        voxel_size: Voxel edge length in metres (e.g. 0.2).

    Returns:
        Tuple of (image, voxel_grid, voxels).
        ``image`` is None if the tile is empty.
    """
    points = np.array(df[["x", "y", "z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()

    x_coords = [v.grid_index[0] for v in voxels]
    y_coords = [v.grid_index[1] for v in voxels]

    if not x_coords or not y_coords:
        print("Voxel projection: tile is empty, skipping.")
        return None, voxel_grid, voxels

    image_size = max(max(x_coords), max(y_coords)) + 1
    image = np.zeros((image_size, image_size))

    for x, y in zip(x_coords, y_coords):
        image[x, y] = 255

    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

    return image, voxel_grid, voxels


def hough_transform(
    image: np.ndarray,
    kernel_size: int = 7,
    blur_sigma: int = 190,
    canny_low: int = 0,
    canny_high: int = 100,
    rho: float = 1.0,
    theta_deg: float = 1.0,
    vote_threshold: int = 8,
    min_line_length: int = 15,
    max_line_gap: int = 1,
) -> tuple[int, list[float]]:
    """Apply a probabilistic Hough line transform to a binary top-down image.

    Steps:
        1. Gaussian blur to suppress noise.
        2. Canny edge detection.
        3. Probabilistic Hough transform (HoughLinesP).

    Args:
        image:           2-D numpy array from ``voxel2image``.
        kernel_size:     Gaussian blur kernel size (must be odd).
        blur_sigma:      Gaussian blur sigma value.
        canny_low:       Lower Canny hysteresis threshold.
        canny_high:      Upper Canny hysteresis threshold.
        rho:             Hough grid distance resolution (pixels).
        theta_deg:       Hough grid angular resolution (degrees).
        vote_threshold:  Minimum votes to accept a line.
        min_line_length: Minimum line length in pixels.
        max_line_gap:    Maximum gap between line segments in pixels.

    Returns:
        Tuple of (number_of_lines_detected, list_of_line_lengths).
        Returns (0, []) when no lines are found.
    """
    blur_gray = cv2.GaussianBlur(image, (kernel_size, kernel_size), blur_sigma)

    plt.imshow(blur_gray, cmap="gray")
    plt.axis("off")
    plt.show()

    img_uint8 = np.uint8(blur_gray)
    edges = cv2.Canny(img_uint8, canny_low, canny_high)

    plt.imshow(edges)
    plt.axis("off")
    plt.show()

    theta = np.pi / 180 * theta_deg
    line_canvas = np.zeros_like(image)

    lines = cv2.HoughLinesP(
        edges, rho, theta, vote_threshold,
        np.array([]), min_line_length, max_line_gap,
    )

    if lines is None:
        print("No Hough lines detected.")
        return 0, []

    line_lengths = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_canvas, (x1, y1), (x2, y2), (255, 0, 0), 5)
            length = np.sqrt((y2 - y1) ** 2 - (x2 - x1) ** 2)
            line_lengths.append(length)

    overlay = cv2.addWeighted(image, 0.8, line_canvas, 1, 0)
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

    print(f"Hough lines detected: {len(lines)}")
    return len(lines), line_lengths

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Basic_Functions import *
from Ground_Filtering import *
from Building_Filtering import * 
from Vegetation_Filtering import *
from sklearn.decomposition import PCA
from scipy.ndimage.filters import sobel
from skimage.feature import canny
from skimage.feature import canny, peak_local_max
from skimage.transform import hough_line, probabilistic_hough_line
import cv2
import numpy as np
import math
from math import sqrt
import joblib



def tile_generator(ply_file, tile_size):
    
    with open(ply_file, 'rb') as f:
        plydata = PlyData.read(f)
        
    vertices = plydata.elements[0]
    
    x_min = np.int32(vertices['x'].min())
    x_max = np.int32(vertices['x'].max())
    num_tiles_x = (x_max - x_min) // tile_size + 1
    x_coordinates = np.linspace(x_min, x_min + tile_size * num_tiles_x, num=num_tiles_x + 1)
    
    y_min = np.int32(vertices['y'].min())
    y_max = np.int32(vertices['y'].max())
    num_tiles_y = (y_max - y_min) // tile_size + 1
    y_coordinates = np.linspace(y_min, y_min + tile_size * num_tiles_y, num=num_tiles_y + 1)
    
    bounding_boxes = []
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            x_min = x_coordinates[i]
            x_max = x_coordinates[i + 1]
            y_min = y_coordinates[j]
            y_max = y_coordinates[j + 1]
            bounding_box = ((x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max))
            bounding_boxes.append(bounding_box)
    
    return bounding_boxes


def generate_3d_grid(df, bounding_boxes, grid_number):
    select_bbox = bounding_boxes[grid_number]
    (x_min, y_min), (_, y_max), (x_max, _), _ = select_bbox
    selected_points = []
    for _, row in df.iterrows():
        x = row['x']
        y = row['y']
        z = row['z']
        sem_class = row['sem_class']
        intensity = row['intensity']
        if x_min <= x <= x_max and y_min <= y <= y_max:
            selected_points.append(row)
        
    new_df = pd.DataFrame.from_records(selected_points)
    new_df.columns = df.columns
    
    return new_df

def filter_dataframes(df, bounding_boxes):
    result_dfs = []
    
    for bbox in bounding_boxes:
        # Extract the bounding box coordinates
        (x_min, y_min), (_, y_max), (x_max, _), _ = bbox
        
        # Filter the DataFrame based on the bounding box
        filtered_df = df[(df['x'] >= x_min) & (df['x'] <= x_max) & (df['y'] >= y_min) & (df['y'] <= y_max)]
        
        
        result_dfs.append(filtered_df)
    
    return result_dfs



def voxel2image(df, voxel_size):
    points = np.array(df[['x','y','z']])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
  

    # Create a voxel grid from the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=voxel_size)
    
    #o3d.visualization.draw_geometries([voxel_grid])
    voxels = voxel_grid.get_voxels()
    
    x_coords = []
    y_coords = []
    for i in range(len(voxels)):
        index = voxels[i].grid_index
        x_coords.append(index[0])
        y_coords.append(index[1])
        
    if not x_coords or not y_coords:
        print("No voxels found. The provided point cloud might be empty.")  
        return None, voxel_grid, voxels
    
    image_size = max(max(x_coords), max(y_coords)) + 1
    
    # Create an empty image
    image = np.zeros((image_size, image_size))
    
   
    

    for i in range(len(x_coords)):
        image[x_coords[i], y_coords[i]] = 255
    
    
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return image, voxel_grid, voxels

def hough_transform(image):

    kernel_size = 7
    blur_gray = cv2.GaussianBlur(image,(kernel_size, kernel_size),190)
    
    plt.imshow(blur_gray, cmap='gray')
    plt.axis('off')
    plt.show()
    
    slice1Copy = np.uint8(blur_gray)
    slice1Copy.shape
    slicecanny = cv2.Canny(slice1Copy,0,100)
    
    plt.imshow(slicecanny)
    plt.axis('off')
    plt.show()
    
    
    
    rho = 1 
    theta = np.pi / 180  
    threshold = 8
    min_line_length = 15  
    max_line_gap = 1  
    line_image = np.copy(image) * 0 
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(slicecanny, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    lines
    
    line_length = []
    
    if lines is None:
        lines = [0]
    
    else:

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                dist = np.sqrt((y2-y1)**2 - (x2-x1)**2)
                line_length.append(dist)
                
                    
        lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
        lines
            
        plt.imshow(lines_edges)
        plt.axis('off')
        plt.show()
        
    
    
    print(len(lines))
    
    return len(lines), line_length


ply_file = "Path to Lidar File"
df = convert_ply_to_df(ply_file)
plot(df)

df_ground_removed = ground_filtering(df)
plot(df_ground_removed)

df_building_removed = building_filter(df_ground_removed)
plot(df_building_removed)

df_vegetation_removed, grid = filter_vegetation(df_building_removed)
plot(df_vegetation_removed)

plt.imshow(grid)

bounding_boxes = tile_generator(ply_file, 50)
result_dfs = filter_dataframes(df_vegetation_removed, bounding_boxes)
result_dfs

line_list = []
final_line_lenght = []

for i in range(0,120):
    image, voxel_grid, voxels = voxel2image(result_dfs[i], 0.2)
    if image is not None:
     line, line_length = hough_transform(image)

     line_list.append(line)
     final_line_lenght.append(line_length)

line_list
final_line_lenght

indexes = [i for i, value in enumerate(line_list) if value > 15] 
select_dfs = []
for i in indexes:
    select_dfs.append(result_dfs[i])
    



powerline_grids = pd.concat(select_dfs)
powerline_grids
plot(powerline_grids)

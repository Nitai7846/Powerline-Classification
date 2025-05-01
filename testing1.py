#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 11:54:29 2023

@author: nitaishah
"""

import combined
from combined import *
import tensorflow as tf
import keras
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
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

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
        
        # Append the filtered DataFrame to the result list
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
    
   
    
    # Set the corresponding pixels to 1 based on the coordinates
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
    
    
    
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 8 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
    max_line_gap = 1  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    
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

ply_file = "/Users/nitaishah/Downloads/DALESObjects/test/5135_54435_new.ply"
df = convert_ply_to_df(ply_file)
plot(df)
df_ground_removed = ground_filtering(df)
df_veg_removed, grid = filter_vegetation(df_ground_removed)
plt.imshow(grid)
plot(df_veg_removed)

bounding_boxes = tile_generator(ply_file, 50)
result_dfs = filter_dataframes(df_ground_removed, bounding_boxes)
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

indexes = [i for i, value in enumerate(line_list) if value > 4]
select_dfs = []
for i in indexes:
    select_dfs.append(result_dfs[i])



scaler = joblib.load('/Users/nitaishah/Desktop/Imp LiDAR/new_weights/scaler_file.pkl')
model = keras.models.load_model('/Users/nitaishah/Desktop/Imp LiDAR/new_weights/model.h5')

powerline_dfs = []
powerline_dfs_new = []
veg_dfs = []

for i in range(len(select_dfs)):
    if select_dfs[i].shape[0] > 250:
        trial_df, ind, dist  = kdtree_point(select_dfs[i])
        trial_df = create_features(trial_df)
        feature_list = ['z','intensity','distance','surface_variation','linearity','planarity','scattering','anistropy','curvature','sum_of_eig','omnivariance','eigentropy']
        model_data = trial_df[feature_list]
        
    
        
        model_data = scaler.transform(model_data)
        predicted = model.predict(model_data)
                
        final_labels = tf.argmax(predicted, axis=1).numpy()
        trial_df['final_labels'] = final_labels 
            
        powerline_points = trial_df[trial_df['final_labels']==1]
        powerline_dfs.append(powerline_points)
        
        veg_points =  trial_df[trial_df['final_labels']==0]
        veg_dfs.append(veg_points)





powerline_final = pd.concat(powerline_dfs)
veg_final = pd.concat(veg_dfs)
plot(powerline_final)    
plot(veg_final)
plot(select_dfs[10])
plot(df_ground_removed)
    
result_dfs[0]


for i in range(len(result_dfs)):
    if result_dfs[i].shape[0] > 250:
        trial_df, ind, dist  = kdtree_point(result_dfs[i])
        trial_df = create_features(trial_df)
        feature_list = ['z','intensity','distance','surface_variation','linearity','planarity','scattering','anistropy','curvature','sum_of_eig','omnivariance','eigentropy']
        model_data = trial_df[feature_list]
        
    
        
        model_data = scaler.transform(model_data)
        predicted = model.predict(model_data)
                
        final_labels = tf.argmax(predicted, axis=1).numpy()
        trial_df['final_labels'] = final_labels 
            
        powerline_points = trial_df[trial_df['final_labels']==1]
        powerline_dfs_new.append(powerline_points)
        
        veg_points =  trial_df[trial_df['final_labels']==0]
        veg_dfs.append(veg_points)

powerline_final_new = pd.concat(powerline_dfs_new)
plot(powerline_final_new)    

veg_final = pd.concat(veg_dfs)
plot(veg_final)

plot(trial_df[trial_df['final_labels']==0])
    





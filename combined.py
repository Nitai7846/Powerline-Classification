#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:43:54 2023

@author: nitaishah
"""

import csv
import plyfile
import open3d as o3d
from plyfile import PlyData, PlyElement
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import CSF
from sklearn.neighbors import KDTree
import math
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import joblib
import plotly.graph_objects as go

def plot(df):
    points = np.array(df[['x','y','z']])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    
    
def ply_to_csv(ply, csv):
    with open(ply, 'rb') as f:
          plydata = PlyData.read(f)


    plydata.elements[0].name


    vertices = plydata.elements[0]

    plydata.elements[0].properties


    with open(csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['x', 'y', 'z', 'intensity', 'sem_class', 'ins_class'])
        # Write vertex data
        for vertex in vertices:
            writer.writerow([vertex[0], vertex[1], vertex[2], vertex[3], vertex[4], vertex[5]])



    
def convert_ply_to_df(ply_file):
    with open(ply_file, 'rb') as f:
        plydata = PlyData.read(f)

    vertices = plydata.elements[0]

    data = {
        'x': vertices['x'],
        'y': vertices['y'],
        'z': vertices['z'],
        'intensity': vertices['intensity'],
        'sem_class': vertices['sem_class'],
        'ins_class': vertices['ins_class']
    }

    dataframe = pd.DataFrame(data)

    return dataframe 

    
def ground_filtering(df):
    
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = 0.5
    print("Pre filtering the number of point's are : ", len(df))
    points = np.array(df[['x', 'y', 'z', 'intensity', 'sem_class']])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    csf.setPointCloud(pcd.points)
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)
    df = pd.DataFrame(points[non_ground]) 
    df.columns = ['x', 'y', 'z', 'intensity', 'sem_class'] 
    print("Post filtering the number of point's are : ", len(df))
    return df 

def kdtree_point(df):
    x = np.array(df['x'])
    y = np.array(df['y'])
    z = np.array(df['z'])
    X = np.vstack((x,y,z)).T

    tree = KDTree(X, leaf_size=2)
    dist, ind = tree.query(X, k=250)
    curvature_cols = []
    lin_cols = []
    PC1_cols = []
    PC2_cols = []
    PC3_cols = []
    lambda0_cols = []
    lambda1_cols = []
    lambda2_cols = []
    distance = []
    
    for i in range(0, len(df)):

        new = np.cov(X[ind[i][1:]].T)
        
        eig_vals, eig_vecs = np.linalg.eig(new)
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,idx]
        
        PC1, PC2, PC3 = eig_vecs.T
        lambda0, lambda1, lambda2 = eig_vals
        
        PC1_cols.append(PC1)
        PC2_cols.append(PC2)
        PC3_cols.append(PC3)
        lambda0_cols.append(lambda0)
        lambda1_cols.append(lambda1)
        lambda2_cols.append(lambda2)
        distance.append(dist[i][249])
    df['PC1'] = PC1_cols
    df['PC2'] = PC2_cols
    df['PC3'] = PC3_cols
    df['eig_val_0'] = lambda0_cols
    df['eig_val_1'] = lambda1_cols
    df['eig_val_2'] = lambda2_cols
    df['distance'] = distance
    df['e0'] = (df['eig_val_0'])/(df['eig_val_0'] + df['eig_val_1']+ df['eig_val_2'])
    df['e1'] = (df['eig_val_1'])/(df['eig_val_0'] + df['eig_val_1']+ df['eig_val_2'])
    df['e2'] = (df['eig_val_2'])/(df['eig_val_0'] + df['eig_val_1']+ df['eig_val_2'])
    
    return df, ind, dist 


def create_features(df):
    df['surface_variation'] = (df['eig_val_2'])/(df['eig_val_0'] + df['eig_val_1'] + df['eig_val_2'])
    df['linearity'] = (df['e0'] - df['e1'])/(df['e0'])
    df['planarity'] = (df['e1'] - df['e2'])/(df['e0'])
    df['scattering'] = (df['e2'])/(df['e0'])
    df['anistropy'] = (df['e0'] - df['e2'])/(df['e1'])
    df['curvature'] = (df['eig_val_0'])/(df['eig_val_0'] + df['eig_val_1'] + df['eig_val_2'])
    df['sum_of_eig'] = (df['eig_val_0'] + df['eig_val_1'] + df['eig_val_2'])
    df['omnivariance'] = pow((df['e0']*df['e1']*df['e2']), 1/3)
    df['eigentropy'] = df.apply(lambda row: -row['e0']*math.log(row['e0']) + -row['e1']*math.log(row['e1']) + -row['e2']*math.log(row['e2']), axis=1)
    return df

def compute_density(df, radius):
    points = np.array(df[['x','y','z']])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    new_ls = []
    for i in range(df.shape[0]):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        density = (0.75*k)/(3.14 * radius**3)
        new_ls.append(density)
    df['density'] = new_ls
    return df
    


def buiilding_filter(df):
    points = np.array(df[['x','y','z']])
    points_other = np.array(df[['intensity','sem_class']])
    
    original_indexes = np.array(df.index)
    len(original_indexes)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    #o3d.visualization.draw_geometries([pcd])
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([pcd],point_show_normal=True)
    
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=0,
        min_num_points=10,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    
    print("Detected {} patches".format(len(oboxes)))
    
    geometries = []
    for obox in oboxes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(obox.color)
        geometries.append(mesh)
        geometries.append(obox)
    geometries.append(pcd)
    
    #o3d.visualization.draw_geometries(geometries)
    
    pcd_trial = pcd
    oboxes_trial = oboxes
    
    for obox in oboxes_trial:
        indices = obox.get_point_indices_within_bounding_box(pcd_trial.points)
        pcd_trial = pcd_trial.select_by_index(indices, invert=True)
    
    #o3d.visualization.draw_geometries([pcd_trial])
    
   
    
    
    df_new = pd.DataFrame()
    pop_array = np.array(pcd_trial.points)
    pop_array[0]
    
    x_list = []
    y_list = []
    z_list = []
    for i in range(0,pop_array.shape[0]):
        x_list.append(pop_array[i][0])
        y_list.append(pop_array[i][1])
        z_list.append(pop_array[i][2])
        
    df_new['x'] = x_list
    df_new['y'] = y_list
    df_new['z'] = z_list
    
    
    
    
    # List the common columns between df1 and df2
    
    
    # Assuming df1 and df2 are your dataframes
    
    # Merge df2 with df1 based on 'x', 'y', and 'z' columns
    merged_df = pd.merge(df_new, df[['x', 'y', 'z', 'sem_class', 'ins_class', 'intensity']], on=['x', 'y', 'z'], how='left')
    
    return merged_df


def filter_vegetation(df,  cell_size=1.0, density_threshold=10):
    
    points = np.array(df[['x','y','z']])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    point_cloud = np.array(pcd.points)
    # Determine the grid size based on the cell size
    grid_size = int(np.ceil(np.max(point_cloud[:, :2]) / cell_size))
    
    # Create an empty grid
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # Iterate through each point in the point cloud
    for point in point_cloud:
        x, y, _ = point
        
        # Determine the cell indices for the point
        cell_x = int(x / cell_size)
        cell_y = int(y / cell_size)
        
        # Increment the cell value to represent point density
        grid[cell_x, cell_y] += 1
    
    # Iterate through each cell in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Check the cell value against the density threshold
            if grid[i, j] > density_threshold:
                # Check the neighboring cells for high-density vegetation
                neighbor_count = np.sum(grid[max(i-1, 0):min(i+2, grid_size), max(j-1, 0):min(j+2, grid_size)] > density_threshold)
                
                # Remove the cell if it fails the neighboring cells criterion
                if neighbor_count > 3:
                    grid[i, j] = 0
    
    # Create a mask to filter out non-vegetation points
    mask = np.zeros(len(point_cloud), dtype=bool)
    
    # Iterate through each point in the point cloud
    for idx, point in enumerate(point_cloud):
        x, y, _ = point
        
        # Determine the cell indices for the point
        cell_x = int(x / cell_size)
        cell_y = int(y / cell_size)
        
        # Check the grid value for the corresponding cell
        if grid[cell_x, cell_y] > 0:
            mask[idx] = True
    
    # Return the filtered point cloud
    filtered_cloud = point_cloud[mask]
    
    

    filtered_cloud
    
    
    df_new = pd.DataFrame()
    
     
    
    x_list = []
    y_list = []
    z_list = []
    for i in range(0,filtered_cloud.shape[0]):
        x_list.append(filtered_cloud[i][0])
        y_list.append(filtered_cloud[i][1])
        z_list.append(filtered_cloud[i][2])
        
    
    df_new['x'] = x_list
    df_new['y'] = y_list
    df_new['z'] = z_list
    
    
    
    merged_df = pd.merge(df_new, df[['x', 'y', 'z', 'sem_class',  'intensity']], on=['x', 'y', 'z'], how='left')
    return merged_df, grid


def select_vegetation(df,  cell_size=1.0, density_threshold=10):
    
    points = np.array(df[['x','y','z']])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    point_cloud = np.array(pcd.points)
    # Determine the grid size based on the cell size
    grid_size = int(np.ceil(np.max(point_cloud[:, :2]) / cell_size))
    
    # Create an empty grid
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # Iterate through each point in the point cloud
    for point in point_cloud:
        x, y, _ = point
        
        # Determine the cell indices for the point
        cell_x = int(x / cell_size)
        cell_y = int(y / cell_size)
        
        # Increment the cell value to represent point density
        grid[cell_x, cell_y] += 1
    
    # Iterate through each cell in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Check the cell value against the density threshold
            if grid[i, j] > density_threshold:
                # Check the neighboring cells for high-density vegetation
                neighbor_count = np.sum(grid[max(i-1, 0):min(i+2, grid_size), max(j-1, 0):min(j+2, grid_size)] > density_threshold)
                
                # Remove the cell if it fails the neighboring cells criterion
                if neighbor_count < 3:
                    grid[i, j] = 0
    
    # Create a mask to filter out non-vegetation points
    mask = np.zeros(len(point_cloud), dtype=bool)
    
    # Iterate through each point in the point cloud
    for idx, point in enumerate(point_cloud):
        x, y, _ = point
        
        # Determine the cell indices for the point
        cell_x = int(x / cell_size)
        cell_y = int(y / cell_size)
        
        # Check the grid value for the corresponding cell
        if grid[cell_x, cell_y] > 0:
            mask[idx] = True
    
    # Return the filtered point cloud
    filtered_cloud = point_cloud[mask]
    
    

    filtered_cloud
    
    
    df_new = pd.DataFrame()
    
     
    
    x_list = []
    y_list = []
    z_list = []
    for i in range(0,filtered_cloud.shape[0]):
        x_list.append(filtered_cloud[i][0])
        y_list.append(filtered_cloud[i][1])
        z_list.append(filtered_cloud[i][2])
        
    
    df_new['x'] = x_list
    df_new['y'] = y_list
    df_new['z'] = z_list
    
    
    
    merged_df = pd.merge(df_new, df[['x', 'y', 'z', 'sem_class',  'intensity']], on=['x', 'y', 'z'], how='left')
    return merged_df, grid



def reassign_labels(df):
    # Calculate number of points in each semantic class before replacing
    class_counts_before = df['sem_class'].value_counts().sort_index()
    print("Number of points in each semantic class before replacing:")
    print(class_counts_before)

    # Define the semantic class mapping
    sem_class_map = {2: 0, 3: 1, 5: 2, 6: 3, 7: 4, 8: 5}

    # Replace the semantic class labels
    df['sem_class'] = df['sem_class'].replace(sem_class_map)

    # Sort the DataFrame by the updated 'sem_class' column
    df.sort_values(by=['sem_class'], inplace=True)

    # Calculate the number of points in each semantic class after replacing
    class_counts_after = df['sem_class'].value_counts().sort_index()
    print("Number of points in each semantic class after replacing:")
    print(class_counts_after)

    return df



def create_dataframe(df):
    df_model = pd.DataFrame()
    df_model['x'] = df['x']
    df_model['y'] = df['y']
    df_model['z'] = df['z']
    df_model['curvature'] = df['curvature']
    df_model['linearity'] = df['linearity']
    df_model['planarity'] = df['planarity']
    df_model['scattering'] = df['scattering']
    df_model['omnivariance'] = df['omnivariance']
    df_model['eigentropy'] = df['eigentropy']
    df_model['intensity'] = df['intensity']
    df_model['sem_class'] = df['sem_class']
    return df_model

def vis_split(df, feature_list, target_list):
    X = df[feature_list]
    y = df[target_list]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = pd.DataFrame(X_train)
    X_train.columns = X.columns
    X_test.columns = X.columns
    return X_train, X_test

def custom_train_test(df, feature_list, target_list):
    X = df[feature_list]
    y = df[target_list]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    yhot = np_utils.to_categorical(y)
    yhot_train = np_utils.to_categorical(y_train)
    yhot_test = np_utils.to_categorical(y_test)
    return X_train, X_test,yhot, yhot_train, yhot_test
    
def apply_scaling(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open('scaler_file.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    return X_train, X_test
    

def preprocess_test_dataframe(df, feature_list, target_list, scaler_file):
    X = df[feature_list]
    y = df[target_list]
    with open(scaler_file, 'rb') as f:
      scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    yhot = np_utils.to_categorical(y)
    return X_scaled, yhot



def construct_local_neighborhood(points, radius, k):
    # Remove invalid values (infs and NaNs) from the points array
    valid_indices = np.isfinite(points).all(axis=1)
    points = points[valid_indices]

    # Create a KDTree from the valid points
    kdtree = KDTree(points)

    local_neighborhood = []

    for point in points:
        # Query the KDTree to find the neighbors within the specified radius
        indices = kdtree.query_radius([point], r=radius)[0]

        # If the number of neighbors exceeds k, keep the k nearest neighbors
        if len(indices) > k:
            indices = indices[:k]

        # Add the neighbors to the local neighborhood
        neighborhood = points[indices]
        local_neighborhood.append(neighborhood)

    return local_neighborhood




def calculate_eigenvalues(local_neighborhood):
    PC1_cols = []
    PC2_cols = []
    PC3_cols = []
    lambda0_cols = []
    lambda1_cols = []
    lambda2_cols = []

    for neighborhood in local_neighborhood:
        new = np.cov(neighborhood.T)

        eig_vals, eig_vecs = np.linalg.eig(new)
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,idx]

        PC1, PC2, PC3 = eig_vecs.T
        lambda0, lambda1, lambda2 = eig_vals

        PC1_cols.append(PC1)
        PC2_cols.append(PC2)
        PC3_cols.append(PC3)
        lambda0_cols.append(lambda0)
        lambda1_cols.append(lambda1)
        lambda2_cols.append(lambda2)

    return PC1_cols, PC2_cols, PC3_cols, lambda0_cols, lambda1_cols, lambda2_cols


def kdtree_with_eigenvalues(df, radius, k):
    x = np.array(df['x'])
    y = np.array(df['y'])
    z = np.array(df['z'])
    points = np.vstack((x, y, z)).T

    local_neighborhood = construct_local_neighborhood(points, radius, k)

    PC1_cols, PC2_cols, PC3_cols, lambda0_cols, lambda1_cols, lambda2_cols = calculate_eigenvalues(local_neighborhood)

    df['PC1'] = PC1_cols
    df['PC2'] = PC2_cols
    df['PC3'] = PC3_cols
    df['eig_val_0'] = lambda0_cols
    df['eig_val_1'] = lambda1_cols
    df['eig_val_2'] = lambda2_cols
    df['e0'] = df['eig_val_0'] / (df['eig_val_0'] + df['eig_val_1'] + df['eig_val_2'])
    df['e1'] = df['eig_val_1'] / (df['eig_val_0'] + df['eig_val_1'] + df['eig_val_2'])
    df['e2'] = df['eig_val_2'] / (df['eig_val_0'] + df['eig_val_1'] + df['eig_val_2'])

    return df


def visualize_local_neighborhood(local_neighborhood):
    fig = go.Figure()

    for neighborhood in local_neighborhood:
        x = neighborhood[:, 0]
        y = neighborhood[:, 1]
        z = neighborhood[:, 2]

        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers'))

    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()


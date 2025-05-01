#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Basic_Functions import *


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









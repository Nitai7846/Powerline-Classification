#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Basic_Functions import *


def building_filter(df):
    points = np.array(df[['x','y','z']])
    points_other = np.array(df[['intensity','sem_class']])
    
    original_indexes = np.array(df.index)
    len(original_indexes)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
   
    
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
    
    
    
    pcd_trial = pcd
    oboxes_trial = oboxes
    
    for obox in oboxes_trial:
        indices = obox.get_point_indices_within_bounding_box(pcd_trial.points)
        pcd_trial = pcd_trial.select_by_index(indices, invert=True)
    
   
    
    
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
    
    
    
    
    merged_df = pd.merge(df_new, df[['x', 'y', 'z', 'sem_class', 'intensity']], on=['x', 'y', 'z'], how='left')
    
    return merged_df


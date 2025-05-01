#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Basic_Functions import *


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

#ply_file = "ply file path"
#df = convert_ply_to_df(ply_file)
#plot(df)

#df_ground_removed = ground_filtering(df)
#plot(df_ground_removed)

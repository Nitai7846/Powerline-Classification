#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

def plot(df):
    points = np.array(df[['x','y','z']])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from Basic_Functions import *
from Ground_Filtering import *
from Building_Filtering import * 
from Vegetation_Filtering import *



ply_file = "path to LiDAR file"
df = convert_ply_to_df(ply_file)
plot(df)

df_ground_removed = ground_filtering(df)
plot(df_ground_removed)

df_building_removed = building_filter(df_ground_removed)
plot(df_building_removed)

df_vegetation_removed, grid = filter_vegetation(df_building_removed)
plot(df_vegetation_removed)



# Original Data Points 
print(df.shape[0])

# Data Points after applying all 3 filters
print(df_vegetation_removed.shape[0]) 

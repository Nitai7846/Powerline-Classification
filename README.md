Powerline Classification from LiDAR Data using Deep Learning

This project focuses on identifying powerline points from raw LiDAR data using a hybrid approach combining traditional computer vision (Hough Transform) and deep learning techniques. The pipeline automates powerline inspection, essential for maintaining electricity transmission infrastructure.

Project Overview

The increasing use of drones with LiDAR sensors enables efficient inspection of powerlines spanning remote and rugged terrains. This project develops an end-to-end pipeline that:

Processes raw LiDAR .ply files
Applies filtering techniques to remove ground, buildings, and vegetation
Detects powerline candidate regions using Hough Transform
Classifies powerline points using a trained neural network model
Outputs a cleaned point cloud with labeled powerline structures

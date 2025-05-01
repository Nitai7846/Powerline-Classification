#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 11:21:01 2023

@author: nitaishah
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import sklearn
from sklearn.preprocessing import OneHotEncoder
from Preprocessing import combined
from Preprocessing.combined import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix




df_vegetation = pd.read_csv("/Users/nitaishah/Desktop/Imp LiDAR/combining/Vegetation_Merged.csv")
df_powerline = pd.read_csv("/Users/nitaishah/Desktop/Imp LiDAR/combining/Powerline_Merged.csv")
df_pole = pd.read_csv("/Users/nitaishah/Desktop/Imp LiDAR/combining/Pole_Merged.csv")
df_building = pd.read_csv("/Users/nitaishah/Desktop/Imp LiDAR/combining/Building_Merged.csv")
merged_df = pd.concat([df_vegetation, df_powerline, df_pole, df_building], axis=0)
plot(merged_df)

#desired_classes = [2, 5, 7, 8]
#df = df[df['sem_class'].isin(desired_classes)]


def replace_label(df):
    sem_class_map = {2: 0, 5: 1, 7: 2, 8: 3}
    df['sem_class'] = df['sem_class'].replace(sem_class_map)
    return df

def drop_rows_with_nan(df):
    return df.dropna(subset=None, how='any')



def select_features(df, feature_list, target_list):
    X = df[feature_list]
    y = df[target_list]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #indices = X_test.index
    return X_train, X_test, y_train, y_test

def vis_test(merged_df, X_test):
    merged_df = merged_df.reset_index(drop=True)
    vis_df = merged_df.loc[X_test.index]
    return vis_df
    
    
def apply_encoding(y_train, y_test):
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return y_train, y_test

def apply_scaling(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open('scaler_file.pkl', 'wb') as file:
        pickle.dump(scaler, file)
    return X_train, X_test

def convert_probability(predicted, X_test, y_test):
    predicted_labels = tf.argmax(predicted, axis=1).numpy()
    test_df = pd.DataFrame({'X_test': X_test, 'y_test': y_test, 'y_pred': predicted_labels})
    return test_df

    



merged_df = replace_label(merged_df)
merged_df = drop_rows_with_nan(merged_df)
df_building.columns
feature_list = ['z','intensity','distance','surface_variation','linearity','planarity','scattering','anistropy','curvature','sum_of_eig','omnivariance','eigentropy']
target_list = ['sem_class']

X_train, X_test, y_train, y_test = select_features(merged_df, feature_list, target_list)
vis_df = vis_test(merged_df, X_test)
y_train, y_test = apply_encoding(y_train, y_test)
X_train, X_test = apply_scaling(X_train, X_test)



def build_model():
    model = keras.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(12,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(4, activation='softmax')) 
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

model = build_model()
model.summary()


history = model.fit(X_train, y_train, epochs=10)

y_pred = model.predict(X_test)
y_pred



predicted_labels = tf.argmax(y_pred, axis=1).numpy()
vis_df.columns
plot(vis_df[vis_df['sem_class']==1])
vis_df['predicted'] = predicted_labels
plot(vis_df[vis_df['predicted']==1])



print(classification_report(vis_df['sem_class'], vis_df['predicted']))

model.save('model.h5')




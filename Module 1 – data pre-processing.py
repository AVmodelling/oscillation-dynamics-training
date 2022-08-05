# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:18:13 2022

@author: Rowan Davies
"""

import os 
import pandas as pd
working_folder = 'C:/Users/cvrd6/Desktop/Rowan/PhD/Conferences and presenations/TRB_2023/Submitted/Paper/Code/ZZZ_test project directory'
os.chdir(working_folder)
path = working_folder+'/01_data training'
if os.path.exists(path) == False:
    os.mkdir(path)
    
#all following vehicles
cars = [2,3,4,5]


#Initialise max and min variables for normalisation
max_speed = 0
min_speed = 1000
max_dist = 0
min_dist = 1000

for p in (1,2,5,6,7,8,9):
    if p < 3:
      file = ('ASta_050719_platoon'+str(p)+'.csv') 
      train_data = pd.read_csv(working_folder+'/'+file, skiprows=5)
    else:
      file = ('ASta_040719_platoon'+str(p)+'.csv') 
      train_data = pd.read_csv(working_folder+'/'+file, skiprows=5) 
    max_speed_new = train_data[['Speed1','Speed2','Speed3','Speed4','Speed5']].max(axis=1).max(axis=0)
    min_speed_new = train_data[['Speed1','Speed2','Speed3','Speed4','Speed5']].min(axis=1).min(axis=0)
    max_dist_new = train_data[['IVS1','IVS2','IVS3','IVS4']].max(axis=1).max(axis=0)
    min_dist_new = train_data[['IVS1','IVS2','IVS3','IVS4']].min(axis=1).min(axis=0)
    if max_speed_new > max_speed:
        max_speed = max_speed_new
    if min_speed_new < min_speed:
        min_speed = min_speed_new
    if max_dist_new > max_dist:
        max_dist = max_dist_new
    if min_dist_new < min_dist:
        min_dist = min_dist_new


for p in (1,2,5,6,7,8,9):
    if p < 3:
      file = ('ASta_050719_platoon'+str(p)+'.csv') 
      train_data = pd.read_csv(working_folder+'/'+file, skiprows=5)
    else:
      file = ('ASta_040719_platoon'+str(p)+'.csv') 
      train_data = pd.read_csv(working_folder+'/'+file, skiprows=5) 

    for c in cars: 
        distance_f = 'IVS'+str(c-1)
        speed_f = 'Speed'+str(c)
        speed_l = 'Speed'+str(c-1)
        df = train_data[['Time', distance_f, speed_f, speed_l, speed_f]]
        df = df.set_axis(['time', 'distance_f', 'speed_f', 'speed_l', 'predictor'], axis=1, inplace=False)
        #df.to_csv('df_orig_platoon'+str(p)+'_car'+str(c)+'_speed.csv')
        df_norm = df
        df_norm['distance_f']=(df['distance_f']-min_dist)/(max_dist-min_dist)
        df_norm['speed_f']=(df['speed_f']-min_speed)/(max_speed-min_speed)
        df_norm['speed_l']=(df['speed_l']-min_speed)/(max_speed-min_speed)
        df_norm['predictor']=(df['predictor']-min_speed)/(max_speed-min_speed)
        df_norm.to_csv(path+'/df_norm_platoon'+str(p)+'_car'+str(c)+'_speed.csv')

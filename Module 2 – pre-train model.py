"""
Created on Mon Aug  1 15:18:13 2022

@author: Rowan Davies
"""

#pip install torchvision
#pip install ssqueezepy
#!pip install timm --no-deps --quiet
#pip install PyWavelets
#!pip3 install pytorch_wavelets

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

working_folder = 'C:/Users/cvrd6/Desktop/Rowan/PhD/Conferences and presenations/TRB_2023/Submitted/Paper/Code/ZZZ_test project directory'
os.chdir(working_folder)

def df_to_tensor(df):
    return torch.from_numpy(df.values).float()

def x_train (data, lookback):
    #data = x_data
    data = data.to_numpy()    
    #data = x_data.iloc[:, 0:3]
    i = lookback
    rows = range(i,data.shape[0])
    samples = np.zeros((len(rows),lookback,data.shape[-1]))
    for j in range(len(rows)):
        indices = np.arange(rows[j] - lookback,rows[j], 1).tolist()
        samples[j,:,:] = data[indices,]
    return samples

def y_train (data, lookback):
    #data = x_data
    targets = data.iloc[:, 1]
    targets.astype('int')
    targets = targets[(lookback):len(data)]
    targets = np.array(targets)
    return targets

def loss_function_classical(x, y):
    optimizer.zero_grad()
    output = net(x)
    output = torch.squeeze(output)
    loss = loss_MSE(output, y)
    loss.backward()
    optimizer.step()
    return(loss)

def train_function_classical(platoons, cars, num_epochs, lookback, batchsize):  
    for p in platoons:
        for c in cars:
            file = "df_norm_platoon"+str(p)+"_car"+str(c)+"_speed.csv"
            path = working_folder+'/01_data training/'+file
            train_data = pd.read_csv(path)
            train_x = pd.DataFrame(train_data, columns=['distance_f', 'speed_f', 'speed_l'])
            #make sure all distances are corrected - this will change the ground truth data.
# =============================================================================
#             for i in range(1,train_x.shape[0]):
#                 train_x['distance_f'].iloc[i] = train_x['distance_f'].iloc[i-1] + \
#                 (train_x['speed_l'].iloc[i-1] - train_x['speed_f'].iloc[i-1])*0.1
#    
# =============================================================================
            for epoch in range (num_epochs):    
                for n in np.arange(0, train_x.shape[0], batchsize).tolist():
                    if (train_x.shape[0]-n < batchsize):
                      batch_size = train_x.shape[0]-n
                    else:
                      batch_size = batchsize
                    x_data = train_x.iloc[n:(batch_size+n),]
                    x_temp = x_train(x_data, lookback)
                    x = torch.from_numpy(x_temp)
                    y_temp = y_train(x_data, lookback)
                    y = torch.from_numpy(y_temp)
                    x = x.float()
                    y = y.float() 
                    #select loss function
                    loss = loss_function_classical(x,y)
                    print(loss.item())
                    tot_loss.append(loss.item())
    return(tot_loss)

batch_size = 1000
learning_rate = 0.001

input_size = 3
sequence_length = 20
hidden_size = 64
num_layers = 2

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0,c0))  
        out = out[:, -1, :]
        out = self.fc(out)
        return out

net = RNN(input_size, hidden_size, num_layers)

# Loss and optimizer
#loss_MSE = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

num_epochs = 5
platoons = [1,2,5,6,7,8]
cars = [2,3,4,5]
p_variable = 'speed'
lookback = 20
batchsize = 1000

loss_MSE = torch.nn.MSELoss()
tot_loss = []
tot_loss = train_function_classical(platoons, cars, num_epochs, lookback, batchsize)
plt.plot(tot_loss)
               
#save model in python
path = working_folder+'/02_LSTM pre trained'
if os.path.exists(path) == False:
    os.mkdir(path)
torch.save(net, path+'pre-trained model_model_PT.pt')
torch.save(net.state_dict(), path+'pre-trained model_net_state_dict')
torch.save(optimizer.state_dict(), path+'pre-trained model_opt_state_dict')

import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import random
from sklearn import metrics
from collections import Counter
from model import *

inpt = "mock_genomes.npy" #input genotype file in numpy npy format
lr = 0.0001 #learning rate
init_dropout = 0.50 #amount initial random masking of input data
out_dir = "mock_output" #output directory
status_ = "mock_case_control_status.npy" #input case/control status file for each corresponding genotype in genotype file
gpu = 1 #number of gpus - only 1 gpu scenario was tested

##Input data preperation and train/test split
df = np.load(inpt, allow_pickle=True)
status = np.load(status_, allow_pickle=True)
status = status.reshape(-1,1)
df = np.concatenate((status, df), axis=1)
df = df.astype(int)
df_case = df[df[:,0]==1].copy()
df_control = df[df[:,0]==0].copy()
df = np.concatenate((df_control, df_case))
df_labels = df[:,0]
df_labels = df_labels.astype(float)
df = df[:,1:]

device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")
classifi = Classifier(data_shape=df.shape[1], init_dropout=init_dropout)
if (device.type == 'cuda') and (gpu > 1):
    classifi = nn.DataParallel(classifi, list(range(gpu)))
classifi.to(device)
optimizer = torch.optim.Adam(classifi.parameters(), lr=lr, weight_decay=1e-3)

#Get checkpoints for models trained with different seeds
checkpoints = []
for i in range(10):
    checkpoints.append(f"{out_dir}/s{i}_e49.model")
    
x_test = torch.from_numpy(df)
y_test = torch.from_numpy(df_labels)
x_test = x_test.to(device)
y_test = y_test.to(device)
x_test = torch.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1])).float()
classifi.eval()

MAS_pm_list = []


ix = 0
#For each trained model in previous step, obtain mean attribution score (MAS) over multiple samples
for check in checkpoints:
    print(ix)
    torch.cuda.empty_cache()
    checkpoint = torch.load(check)

    classifi.load_state_dict(checkpoint['classifi'])
    optimizer.load_state_dict(checkpoint['optimizer'])    
    
    with torch.no_grad():
        y_hat_test = classifi(x_test)
    
    np.random.seed(ix)
    random_pos_val = np.random.randint(low=-1, high=2, size=(x_test.shape[0],))

    dist_list = []
    temp1 = y_hat_test.flatten().detach().cpu().numpy()
    
    for i in range(df.shape[1]):
        x_test_temp = x_test.clone()
        x_test_temp[:,:,i] = torch.tensor(random_pos_val).view(random_pos_val.shape[0], 1)
        with torch.no_grad():
            y_hat_test1 = classifi(x_test_temp)   

        temp2 = y_hat_test1.flatten().detach().cpu().numpy()
        dist = np.mean(np.abs(temp1 - temp2))
        dist_list.append(dist)
    
    MAS_pm_list.append(dist_list)
    ix += 1
    
MAS_pm_list = np.array(MAS_pm_list)
#Save MAS obtained via PM approach
np.save(f"{out_dir}/MAS_pm_list.npy", MAS_pm_list)
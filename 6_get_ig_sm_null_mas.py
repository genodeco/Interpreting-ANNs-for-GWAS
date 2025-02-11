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
from captum.attr import (IntegratedGradients, Saliency)
from model import *

inpt_ = "mock_genomes.npy" #input genotype file in numpy npy format
lr = 0.0001 #learning rate
init_dropout = 0.50 #amount initial random masking of input data
out_dir = "mock_output" #output directory
gpu = 0 #number of gpus - only 1 gpu scenario was tested

device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")
classifi = Classifier(data_shape=df.shape[1], init_dropout=init_dropout)
if (device.type == 'cuda') and (gpu > 1):
    classifi = nn.DataParallel(classifi, list(range(gpu)))
classifi.to(device)
optimizer = torch.optim.Adam(classifi.parameters(), lr=lr, weight_decay=1e-3)


classifi.eval()

MAS_ig_list = []
MAS_sm_list = []


#For each trained model in previous step, obtain mean attribution score (MAS) over multiple samples
for i_check in range(10):
    check = f"{out_dir}/s{i_check}_e49_NULL.model"

    status_ = f"{out_dir}/null_labels_s{seed}.npy"

    ##Input data preperation and train/test split
    df = np.load(inpt_, allow_pickle=True)
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
    
    
    torch.cuda.empty_cache()
    checkpoint = torch.load(check, map_location=device)

    classifi.load_state_dict(checkpoint['classifi'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    inpt = torch.tensor(df_case[:,1:]).clone()
    inpt = inpt.to(device)
    inpt = torch.reshape(inpt, (inpt.shape[0], 1, inpt.shape[1])).float()

    baseline = torch.zeros(inpt.shape[0], 1, inpt.shape[2]).to(device)

    ig = IntegratedGradients(classifi)

    sm = Saliency(classifi)

    attributions, delta = ig.attribute(inpt, baseline, target=None, return_convergence_delta=True, n_steps=100, internal_batch_size=1000)
    attributions = attributions.view(inpt.shape[0], inpt.shape[2])
    attributions = attributions.detach().cpu().numpy()

    attributions_sm = sm.attribute(inpt, target=None)
    attributions_sm = attributions_sm.view(inpt.shape[0], inpt.shape[2])
    attributions_sm = attributions_sm.detach().cpu().numpy()

    attributions = np.abs(attributions)
    attributions_sm = np.abs(attributions_sm)

    for i in range(attributions.shape[0]):  
        attributions[i,:] = attributions[i,:]/np.sum(attributions[i,:])
        attributions_sm[i,:] = attributions_sm[i,:]/np.sum(attributions_sm[i,:])

    MAS_ig = []
    MAS_sm = []
    for i in range(attributions[0,:].shape[0]):
        MAS_ig.append(attributions[:,i].mean())
        MAS_sm.append(attributions_sm[:,i].mean())

    MAS_ig_list.append(MAS_ig)
    MAS_sm_list.append(MAS_sm)


MAS_ig_list = np.array(MAS_ig_list)
MAS_sm_list = np.array(MAS_sm_list)

#Save MAS obtained via IG and SM approaches
np.save(f"{out_dir}/MAS_NULL_ig_list.npy", MAS_ig_list)
np.save(f"{out_dir}/MAS_NULL_sm_list.npy", MAS_sm_list)
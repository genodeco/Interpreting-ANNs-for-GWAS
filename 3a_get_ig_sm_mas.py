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


MAS_ig_list = []
MAS_sm_list = []


#For each trained model in previous step, obtain mean attribution score (MAS) over multiple samples
for check in checkpoints:
    torch.cuda.empty_cache()
    checkpoint = torch.load(check)

    classifi.load_state_dict(checkpoint['classifi'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    with torch.no_grad():
        y_hat_test_cont = classifi(x_test[:df_control.shape[0],:,:])
        y_hat_test_case = classifi(x_test[df_control.shape[0]:,:,:])

    y_hat_test_mask = y_hat_test_case>torch.quantile(y_hat_test_case, 0.70, dim=0, keepdim=True).flatten().item()
    pos_indices = y_hat_test_mask.nonzero()

    y_hat_test_mask = y_hat_test_cont<abs(torch.quantile(-y_hat_test_cont, 0.70, dim=0, keepdim=True).flatten().item())
    neg_indices = y_hat_test_mask.nonzero()

    df_test0 = df[neg_indices[:,0].detach().flatten().cpu().numpy()]
    df_test1 = df[pos_indices[:,0].detach().flatten().cpu().numpy() + df_control.shape[0]]

    inpt = torch.tensor(df_test1).clone()
    inpt = inpt.to(device)
    inpt = torch.reshape(inpt, (inpt.shape[0], 1, inpt.shape[1])).float()

    baseline = torch.zeros(inpt.shape[0], 1, inpt.shape[2]).to(device)

    ig = IntegratedGradients(classifi)

    sm = Saliency(classifi)

    attributions, delta = ig.attribute(inpt, baseline, target=None, return_convergence_delta=True, n_steps=100, internal_batch_size=1000)
    attributions = attributions.view(inpt.shape[0], inpt.shape[2])

    attributions_sm = sm.attribute(inpt, target=None)
    attributions_sm = attributions_sm.view(inpt.shape[0], inpt.shape[2])

    MAS_ig = []
    MAS_sm = []
    for i in range(attributions[0,:].shape[0]):
        MAS_ig.append(attributions[:,i].abs().mean().item())
        MAS_sm.append(attributions_sm[:,i].abs().mean().item())

    MAS_ig_list.append(MAS_ig)
    MAS_sm_list.append(MAS_sm)


MAS_ig_list = np.array(MAS_ig_list)
MAS_sm_list = np.array(MAS_sm_list)

#Save MAS obtained via IG and SM approaches
np.save(f"{out_dir}/MAS_ig_list.npy", MAS_ig_list)
np.save(f"{out_dir}/MAS_sm_list.npy", MAS_sm_list)
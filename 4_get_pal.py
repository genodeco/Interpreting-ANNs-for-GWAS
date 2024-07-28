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

inpt = "mock_genomes.npy" #input genotype file in numpy npy format
out_dir = "mock_output" #output directory
thresh = 99.99 #threshold for assigning significance

#Input data
df = np.load(inpt, allow_pickle=True)
df = df.astype(int)

#Load MAS list from previous script
MAS_list = np.load(f"{out_dir}/MAS_ig_list.npy")
#MAS_list = np.load(f"{out_dir}/MAS_sm_list.npy")
#MAS_list = np.load(f"{out_dir}/MAS_pm_list.npy")

#Min-max scaling
for i in range(MAS_list.shape[0]):
    scaled_data = (MAS_list[i,:] - MAS_list[i,:].min()) / (MAS_list[i,:].max() - MAS_list[i,:].min())  
    MAS_list[i,:] = scaled_data
    
#Get detected positions above threshold for each trained model
MAS_detected = []
for i in range(MAS_list.shape[0]):
    detected = (MAS_list[i,:] > np.percentile(MAS_list[i,:], thresh)).nonzero()[0] 
    MAS_detected.append(detected)
    
MAS_detected_common_LD_signal = MAS_detected.copy()        
MAS_detected = np.array(MAS_detected)  

#Get common signals between models - since there is LD in real data, different detected SNPs by different models might actually be the same signal
for i in range(MAS_detected.shape[0]):
    print(i)
    for i2 in range(MAS_detected.shape[1]):
        for ix in range(-99,100):
            if ix == 0:
                continue
            if MAS_detected[i][i2]+ix >= df.shape[1]:
                continue
            if abs(np.corrcoef(df[:,MAS_detected [i][i2]], df[:,MAS_detected [i][i2]+ix])[0][1]) > 0.5:
                MAS_detected_common_LD_signal[i] = np.append(MAS_detected_common_LD_signal[i], MAS_detected[i][i2]+ix)        
for i in range(len(MAS_detected_common_LD_signal)):
    MAS_detected_common_LD_signal[i] = np.unique(MAS_detected_common_LD_signal[i])
    MAS_detected_common_LD_signal[i] = np.sort(MAS_detected_common_LD_signal[i])      
flattened = [item for sublist in MAS_detected_common_LD_signal for item in sublist]
element_counts = Counter(flattened)

#Get PAL_Common - signals detected by all models
PAL_common = [element for element, count in element_counts.items() if count >= MAS_list.shape[0]]
PAL_common = np.array(PAL_common)
PAL_common = np.sort(PAL_common)

#Calculate Adjusted Mean Attribution Score (AMAS), by weighting mean MAS based on occurence above threshold
mean_MAS = np.mean(MAS_list, axis=0)
detected = (mean_MAS > np.percentile(mean_MAS, thresh)).nonzero()[0]  
AMAS = mean_MAS.copy()
for i in range(len(detected)):
    AMAS[detected[i]] = AMAS[detected[i]] * (element_counts[detected[i]]/MAS_list.shape[0])

#PAL_AMAS consists of detected positions which are above threshold after weighting
PAL_AMAS = (AMAS > np.percentile(mean_MAS, thresh)).nonzero()[0]  

#Manhattan plot of AMAS with colored PAL_AMAS and PAL_Common positions
fig = plt.figure(figsize=(18,6))
plt.scatter(range(len(mean_MAS)), mean_MAS, color='silver')
plt.scatter(PAL_common, [mean_MAS[i] for i in PAL_common], marker='o', s=150, color='cornflowerblue', label='PAL_Common')
plt.scatter(PAL_AMAS, [mean_MAS[i] for i in PAL_AMAS], marker='o', s=50, color='black', label='PAL_AMAS')
plt.axhline(y=np.percentile(mean_MAS, thresh), color='r', linestyle='--')
plt.xlabel('SNPs', fontsize=15)
plt.ylabel('-log10(P)', fontsize=15)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)
fig.savefig(f'{out_dir}/PAL_AMAS_Common.png', format='png', dpi=300)

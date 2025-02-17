import sys
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import halfnorm
import matplotlib.pyplot as plt
from collections import Counter


inpt_ = "mock_genomes.npy" #input genotype file in numpy npy format
df = np.load(inpt_, allow_pickle=True)
df = df.astype(int)
out_dir = "mock_output"
MAS_null = np.load(f"{out_dir}/MAS_NULL_ig_list.npy")
#MAS_null = np.load(f"{out_dir}/MAS_NULL_sm_list.npy")
thresh = 99.99
params = halfnorm.fit(MAS_null.flatten())

fig = plt.figure(figsize=(8, 6))

plt.hist(MAS_null.flatten(), bins=200, density=True, alpha=0.6, color='blue')

# Generate x values for the fitted distribution
x = np.linspace(0, max(MAS_null.flatten()), 100)
pdf = halfnorm.pdf(x, loc=0, scale=params[1])

# Plot the fitted half-normal PDF
plt.plot(x, pdf, 'r-', lw=2)
plt.xlabel('Data')
plt.ylabel('Density')
plt.show()
fig.savefig(f'{out_dir}/half_norm_fit.png', format='png', dpi=300)

# Tail region 
fig = plt.figure(figsize=(8, 8))
threshold = np.percentile(MAS_null.flatten(), 90)  # Define a threshold for the tail
tail_data = MAS_null.flatten()[MAS_null.flatten() >= threshold]  # Filter tail data

# Q-Q Plot
#plt.figure(figsize=(8, 6))
quantiles = np.linspace(0.90, 1, len(tail_data))
observed = np.sort(tail_data)
expected = halfnorm.ppf(quantiles, loc=0, scale=params[1])

plt.scatter(expected, observed)
plt.plot(expected, expected, 'r--', label="Ideal Fit")
plt.xlabel("Expected Quantiles")
plt.ylabel("Observed Quantiles")
plt.title("Q-Q Plot - 90% Tail")
plt.legend()
fig.savefig(f'{out_dir}/half_norm_fit_tail90_QQ.png', format='png', dpi=300)

PAL_AMAS = np.load(f"{out_dir}/PAL_AMAS_ig.npy")
MAS_main = np.load(f"{out_dir}/MAS_ig_list.npy")
AMAS = np.load(f"{out_dir}/AMAS_ig_list.npy")

scale = params[1]

err = []
err_mas = []
boot_num = 100
for boot in range(boot_num):
    print(boot)
    
    MAS_list = halfnorm.rvs(scale=scale, size=(MAS_main.shape[0], MAS_main.shape[1]))

    Array1 = MAS_main
    Array2 = MAS_list
    
    # For each row
    Array2_ranked_corrected = np.zeros_like(Array2)
    for i in range(Array1.shape[0]):
        # Indices that would sort Array1[i] in descending order (from highest to lowest)
        sorted_indices_desc = np.argsort(-Array1[i])
        # Sort Array2[i] in descending order
        sorted_Array2_desc = np.sort(Array2[i])[::-1]
        # Map the highest values to the positions of the highest values in Array1
        Array2_ranked_corrected[i, sorted_indices_desc] = sorted_Array2_desc

    MAS_list = Array2_ranked_corrected


    MAS_detected = []
    for i in range(MAS_list.shape[0]):
        detected = (MAS_list[i,:] > np.percentile(MAS_list[i,:], thresh)).nonzero()[0] 
        MAS_detected.append(detected)
        
    MAS_detected_common_LD_signal = MAS_detected.copy()        
    MAS_detected = np.array(MAS_detected)  
    
    #Get common signals between models - since there is LD in real data, different detected SNPs by different models might actually be the same signal
    for i in range(MAS_detected.shape[0]):
        #print(i)
        for i2 in range(MAS_detected.shape[1]):
            for ix in range(-20,21):
                if ix == 0:
                    continue
                if MAS_detected[i][i2]+ix >= df.shape[1]:
                    continue
                if abs(np.corrcoef(df[:,MAS_detected[i][i2]], df[:,MAS_detected[i][i2]+ix])[0][1]) > 0.5:
                    MAS_detected_common_LD_signal[i] = np.append(MAS_detected_common_LD_signal[i], MAS_detected[i][i2]+ix)        
    for i in range(len(MAS_detected_common_LD_signal)):
        MAS_detected_common_LD_signal[i] = np.unique(MAS_detected_common_LD_signal[i])
        MAS_detected_common_LD_signal[i] = np.sort(MAS_detected_common_LD_signal[i])      
    flattened = [item for sublist in MAS_detected_common_LD_signal for item in sublist]
    element_counts = Counter(flattened)
    
    #Calculate Adjusted Mean Attribution Score (AMAS), by weighting mean MAS based on occurence above threshold
    mean_MAS = np.mean(MAS_list, axis=0)

    
    detected = (mean_MAS > np.percentile(mean_MAS, thresh)).nonzero()[0]  
    AMAS_boot = mean_MAS.copy()
    for i in range(len(detected)):
        AMAS_boot[detected[i]] = AMAS_boot[detected[i]] * (element_counts[detected[i]]/MAS_list.shape[0])

    err_temp = []
    for x in range(len(PAL_AMAS)):
        
        temp_sum = np.sum(AMAS_boot > AMAS[PAL_AMAS[x]])
        err_temp.append(temp_sum)

    err.append(err_temp)

    
err = np.array(err)

p_vals = np.sum(err, axis=0)/(err.shape[0] * MAS_main.shape[1])
np.save(f"{out_dir}/AMAS_pvals.npy", p_vals)

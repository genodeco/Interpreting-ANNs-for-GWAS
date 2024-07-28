import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import normalize
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA

inpt = "mock_genomes.npy" #input genotype file in numpy npy format
out_dir = "mock_output" #output directory
status_ = "mock_case_control_status.npy" #input case/control status file for each corresponding genotype in genotype file

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

#Perform PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(df)
print("Variance explained by each PC:", pca.explained_variance_ratio_)

results = []
for i in range(df.shape[1]):  #Loop over each SNP
    #print(i)
    snp_data = df[:, i]
    X = np.column_stack((snp_data, principal_components))  #Combine SNP data with PCs
    X = sm.add_constant(X)  #Add constant term

    #Fit logistic regression model
    model = sm.Logit(df_labels, X).fit(disp=0)

    results.append({
        'SNP_index': i,
        'p_value': model.pvalues[1],  #p-value for the SNP coefficient
        'inter': model.params[0], #intercept
        'coeff': model.params[1],  #coefficient for the SNP
        'PC1': model.params[2],
        'PC2': model.params[3],
        'PC3': model.params[4],
        'converg': model.mle_retvals['converged']
    })
df_results = pd.DataFrame(results)

df_results.to_csv(f'{out_dir}/logistic_regression.txt', index=False)


p_values = np.array(df_results["p_value"])

sorted_p_values = np.sort(p_values)

#Calculate the expected uniform quantiles
n = len(sorted_p_values)
expected_p_values = np.arange(1, n+1) / (n+1)

#Transform both observed and expected p-values to the -log10 scale
expected_minuslog10p = -np.log10(expected_p_values)
observed_minuslog10p = -np.log10(sorted_p_values)


#Create QQ plot
fig = plt.figure(figsize=(10, 10))
plt.scatter(expected_minuslog10p, observed_minuslog10p, edgecolor='blue', facecolor='none')

#Add identity line
max_val = max(max(expected_minuslog10p), max(observed_minuslog10p))
max_val = max(expected_minuslog10p)
plt.plot([0, max_val], [0, max_val], 'r--', lw=2)

# Set the plot labels and title
plt.xlabel('Expected -log10(P)')
plt.ylabel('Observed -log10(P)')
plt.title('QQ Plot')
plt.axis('square')
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.grid(True, which="both", ls="--", c='gray', alpha=0.5)
fig.savefig(f'{out_dir}/logistic_regression_QQ_plot.pdf', format='pdf')


#Get Bonferroni significance
bonferroni_threshold = 0.05 / len(p_values)

# Calculate -log10 p-values for plotting
neg_log_pvals = -np.log10(p_values)
bonferroni_line = -np.log10(bonferroni_threshold)

fig = plt.figure(figsize=(18,6))
plt.scatter(range(len(neg_log_pvals)), neg_log_pvals, color='silver')
plt.axhline(y=bonferroni_line, color='r', linestyle='--')
plt.xlabel('Genomic Position')
plt.ylabel('-log10(P)')
fig.savefig(f'{out_dir}/logistic_regression_manhattan_plot.pdf', format='pdf')
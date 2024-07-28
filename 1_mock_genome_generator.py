import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Parameters
control_sample_size = 500
genome_size = 20000
causal_pos = 20

control_distribution = [0.10, 0.45, 0.45]
case_distribution = [0.90, 0.05, 0.05]


#Generate case and control mock genomes based on distributions
control_genomes = np.random.choice([-1, 0, 1], size=(control_sample_size, genome_size), p=control_distribution)
case_genomes = np.random.choice([-1, 0, 1], size=(control_sample_size, causal_pos), p=case_distribution)
case_genomes_neutral = np.random.choice([-1, 0, 1], size=(control_sample_size, genome_size - causal_pos), p=control_distribution)
case_genomes = np.hstack((case_genomes, case_genomes_neutral))

all_genomes = np.vstack((control_genomes, case_genomes))

#Perform PCA
pca = PCA(n_components=2)  #2 dimensions
projected = pca.fit_transform(all_genomes)


#Plot PCA
fig = plt.figure(figsize=(6, 6))
plt.scatter(projected[:500, 0], projected[:500, 1], color='red', alpha=0.5, label='Control')
plt.scatter(projected[500:, 0], projected[500:, 1], color='blue', alpha=0.5, label='Case')
plt.title('PCA of mock genomes')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
fig.savefig('mock_genomes_PCA.pdf', format='pdf')
plt.show()

#Save genotypes and phenotypes
status = np.vstack((np.zeros(500), np.ones(500))).flatten()
np.save("mock_case_control_status.npy", status)
np.save("mock_genomes.npy", all_genomes)
# Interpreting-ANNs-for-GWAS

## Description
This is the repository with scripts explained in the manuscript "Interpreting artificial neural networks to detect genome-wide association signals for complex traits".

## Installation

### Prerequisites
- Python 3.9
- Conda (optional, for managing the environment)

### Setting Up the Environment

#### Using Conda (Recommended)
1. **Clone the repository**:
    ```bash
    git clone https://github.com/genodeco/Interpreting-ANNs-for-GWAS.git
    cd Interpreting-ANNs-for-GWAS
    ```

2. **Create a Conda environment**:
    ```bash
    conda create --name my_env python=3.9
    conda activate my_env
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

#### Using Virtualenv
1. **Clone the repository**:
    ```bash
    git clone https://github.com/genodeco/Interpreting-ANNs-for-GWAS.git
    cd Interpreting-ANNs-for-GWAS
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Scripts
The scripts can be run in the following order. You can change the defined variables in the scripts to run with different parameters. Beware that some scripts can take time depending on the computational resources.

```bash
#Generates mock genotypes (mock_genomes.npy) and corresponding phenotype status (mock_case_control_status.npy).
python 1_mock_genome_generator.py

#Performs logistic regression and generates an output file (logistic_regression.txt), a QQ plot (logistic_regression_QQ_plot.pdf) and a Manhattan plot (logistic_regression_manhattan_plot.pdf)
python 2a_logistic_regression.py

#Trains 10 neural networks models with different seeds and outpus trained models (s{seed}_e{epoch}.model), loss (s{seed}_e{epoch}loss.pdf), accuracy (s{seed}_e{epoch}accuracy.pdf) and AUC (s{seed}_e{epoch}auc.pdf) plots.
python 2b_train_model.py

#Calculates mean attribution score -MAS- using integrated gradient (IG) and saliency map (SM) approaches for the trained models and outputs these files (MAS_ig_list.npy, MAS_sm_list.npy).
python 3a_get_ig_sm_mas.py

#Calculates mean attribution score -MAS- using permutation feature importance (PM) approach for the trained models and outputs this file (MAS_pm_list.npy).
python 3b_get_pm_mas.py

#Calculates adjusted mean attribution score -AMAS- and obtains potentially associated loci (PAL) and outputs a Manhattan plot of AMAS with denoted PAL (PAL_AMAS_Common.png). You can change MAS_list variable to define from which method to get PAL (IG, PM or SM).  
python 4_get_pal.py

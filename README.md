# DeepOmicsAE


**Description**

DeepOmicsAE is a specialized repository which uses an autoencoder to extract features from high dimensional multiomics data. The repository provides a comprehensive workflow to train models, optimize parameters, and analyze feature importance in a deep learning context.

**Features**
Data Preprocessing
Autoencoder Model Optimization
Feature Importance Analysis
Customizable Workflow through Jupyter Notebooks

**Prerequisites**
The environment setup is simplified through an environment file, DeepOmicsAE_env.yml.

**Setup**

Clone the repository:

    git clone https://github.com/YourUsername/DeepOmicsAE.git
Navigate to the project directory:

    cd DeepOmicsAE
Create the environment from the .yml file:

    conda env create -f DeepOmicsAE_env.yml
Activate the environment:

 
    conda activate YourEnvironmentName
    
**How to Run**

Open one of the Jupyter notebooks (M01 - expression data pre-processing.ipynb, M02 - DeepOmicsAE model optimization.ipynb, M03a - DeepOmicsAE implementation with custom-optimized parameters.ipynb, M03b - DeepOmicsAE implementation with pre-set parameters.ipynb) to go through different phases of the project.

Run the cells in sequence to perform tasks like data preprocessing, model training, or feature importance analysis.

**File Structure**

.gitignore: Specifies intentionally untracked files to ignore.

DeepOmicsAE_env.yml: Conda environment file.

F01_data_preprocessing_function.py: Functions for data preprocessing.

F02_autoencoder_model_optimization.py: Functions for optimizing the autoencoder model.

F03_Feature_importance_functions.py: Functions for calculating and visualizing feature importance.

LICENSE: License file.

M01 - expression data pre-processing.ipynb: Jupyter notebook for data preprocessing.

M02 - DeepOmicsAE model optimization.ipynb: Jupyter notebook for model optimization.

M03a - DeepOmicsAE implementation with custom-optimized parameters.ipynb: Implementation notebook with custom parameters.

M03b - DeepOmicsAE implementation with pre-set parameters.ipynb: Implementation notebook with pre-set parameters.

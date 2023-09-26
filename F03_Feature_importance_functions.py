#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from sklearn.feature_selection import SelectKBest, f_classif
import random
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def feature_importance_latent(encoder, data, eps):

    """
    This function computes the importance scores for the features of a dataset by perturbing each feature and observing the changes in the latent space of an encoder model.

    Parameters:
    encoder (model): The encoder model used to project data into latent space.
    data (DataFrame): The input data whose features' importance are to be determined.
    eps (float): The perturbation amount added to each feature in the data.

    Output:
    importance_scores (2D numpy array): An array with importance scores for each feature across all 
    dimensions of the latent space.
    """

    data = data.values
    original_latent = encoder.predict(data)[0]
    importance_scores = np.zeros((data.shape[1], original_latent.shape[1]))

    for i in range(data.shape[1]):
        perturbed_data = data.copy()
        perturbed_data[:, i] += eps
        perturbed_latent = encoder.predict(perturbed_data)[0]
        
        # Measure change in latent space
        change = np.mean(np.abs(original_latent - perturbed_latent), axis=0)
        
        importance_scores[i, :] = change

    return importance_scores


# In[3]:


def top_percentile_indices(scores):

    """
    This function calculates the 70th percentile value (threshold) for each column in a 2D array (scores)
    and returns indices of elements that are greater or equal to this threshold. The output is a list 
    of these indices for each column (top_indices), along with the corresponding threshold values (thresholds).
    """
    num_latent_units = scores.shape[1]
    top_indices = []
    thresholds = []
    
    for i in range(num_latent_units):
        threshold = np.percentile(scores[:, i], 70) # 70th percentile value
        indices = np.where(scores[:, i] >= threshold)[0] # indices where score > threshold
        top_indices.append(indices)
        thresholds.append(threshold)
    return top_indices, thresholds


# In[4]:


def plot_density(dfs, thresholds, scaler):
    """
    Function to plot density plots for multiple dataframes in a 2x2 grid and save to a PDF file.
    
    Parameters:
    dfs (list): A list of dataframes.
    thresholds (list): A list of threshold values.
    """
    # Ensure dfs and thresholds have the same length
    assert len(dfs) == len(thresholds), "The lengths of dfs and thresholds must match."

    # Create a figure with 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(8.5/2, 11/2))  # Letter size paper in inches

    # Flatten axs to make it easier to iterate over
    axs = axs.flatten()
    
    # Calculate global min and max
    global_min = min(df['Scaled Importance Values'].min() for df in dfs)
    global_75th_percentile = max(df['Scaled Importance Values'].quantile(0.99) for df in dfs)
    #print(global_75th_percentile)
    
    # Create handles and labels for the legend
    handles, labels = None, None

    for i, df in enumerate(dfs):
        ax = axs[i]
        
        # Plot density plot for red-coded values
        sns.kdeplot(data=df[df['color'] == 'red'], x='Scaled Importance Values', fill=True, color='red', label='Important features', ax=ax)

        # Plot density plot for blue-coded values
        sns.kdeplot(data=df[df['color'] == 'blue'], x='Scaled Importance Values', fill=True, color='blue', label='Other features', ax=ax)

        # Ignore warnings from scikit-learn
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=DataConversionWarning)
        # Add a vertical line for the threshold
        ax.axvline(scaler.transform([[thresholds[i]]])[0,0], color='gray', linestyle='--')

        # Set plot title and labels
        ax.set_title(f'Module {i+1}', fontsize=12)
        ax.set_xlabel('Importance Values', fontsize=12)
        ax.set_ylabel('', fontsize=12)
        
        # Set x-axis limits to global min and max
        ax.set_xlim(global_min, global_75th_percentile)

        # If this is the first plot, save the handles and labels for the legend
        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()


    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5, bottom=0.2)
    
    # Add a single legend for the whole figure
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0))
    
    # Define the PDF output file
    pdf_pages = PdfPages('distribution_important_feature_scores.pdf')

    # Save the figure to the PDF file
    pdf_pages.savefig(fig, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend in the saved figure

    # Close the PdfPages object
    pdf_pages.close()


# In[ ]:


### create a set of dataframes with the important values for each neuron in the latent layer
### normalize the importance values in the range between 0 to 1
### this data will be used to plot the distribution of importance scores

def importance_scores_scaling(importance_scores, thresholds):
    """
    This function scales the importance scores for each feature across all dimensions of latent space, 
    using Min-Max normalization.

    Parameters:
    importance_scores (2D numpy array): An array with unscaled importance scores.
    thresholds (list): A list of threshold values. Each threshold corresponds to a 
    column in 'importance_scores'.
    A score greater than its corresponding threshold gets labeled 'red', otherwise 'blue'.


    Output:
    dfs (list of DataFrames): A list of DataFrames where each DataFrame corresponds to a dimension 
    in the latent space and contains scaled importance scores for each feature.

    Note: The function also adds a color label ('red' or 'blue') for each score based on 
    the threshold values provided as input.
    """
     
    i = importance_scores.shape[1]

    # Create empty DataFrames for each latent unit
    dfs = []
    for i in range(importance_scores.shape[1]):
        df = pd.DataFrame({
            'x': [i+1] * importance_scores.shape[0],
            'Importance Values': importance_scores[:, i]
        })
         # Determine colors based on threshold for each DataFrame
        threshold = thresholds[i]
        df['color'] = ['red' if value > threshold else 'blue' for value in df['Importance Values']]
        dfs.append(df)

    # Access individual DataFrames (e.g., df1, df2, df3, df4)
    df1, df2, df3, df4 = dfs
    combined_df = pd.concat(dfs, ignore_index=True)

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Fit the scaler to 'Importance Values' in the combined dataframe
    scaler.fit(combined_df[['Importance Values']])

    # Apply transformation to each individual dataframe in dfs
    for df in dfs:
        df['Scaled Importance Values'] = scaler.transform(df[['Importance Values']])
    
    return(dfs, scaler)


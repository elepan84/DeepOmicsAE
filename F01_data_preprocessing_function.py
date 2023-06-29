#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial import distance
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[171]:


def data_preprocessing(df, cols_prot, cols_met, cols_clin_binary, cols_clin_cont):
    
    # Preprocesses the expression data in the dataframe.
    
    # Parameters:
    # df (pd.DataFrame): The dataframe containing the data.
    # cols_prot (range): The range of column indices for proteomics data.
    # cols_met (range): The range of column indices for metabolomics data.
    ###### the ranges in cols_prot and cols_met should be contiguous 
    
    # cols_clin_binary (range): The range of column indices for binary clinical data, optimized for sex, 
    ### specified as 0 for female and 1 for male.
    # cols_clin_cont (range): The range of column indices for continous clinical data.
    
    # Returns:
    # result_df: The processed dataframe.
        
    ##################################################################################################
    ##### remove features (columns) that have more than 10% missing data
    threshold = round(df.shape[0] * 0.1)
    nan_counts_per_column = df.isna().sum(axis=0)
    filtered_df = df.loc[:, nan_counts_per_column < threshold]

    # Get the new column positions in the filtered dataframe
    all_indices = [i for i in range(df.shape[1]) if nan_counts_per_column[i] < threshold]
    indices_prot = [i for i in range(cols_prot.start, cols_prot.stop) if nan_counts_per_column[i] < threshold]
    indices_met = [i for i in range(cols_met.start, cols_met.stop) if nan_counts_per_column[i] < threshold]
    indices_clin_bin = [i for i in range(cols_clin_binary.start, cols_clin_binary.stop) if nan_counts_per_column[i] < threshold]
    indices_clin_con = [i for i in range(cols_clin_cont.start, cols_clin_cont.stop) if nan_counts_per_column[i] < threshold]
    new_cols_prot = slice(all_indices.index(indices_prot[0]), all_indices.index(indices_prot[-1])+1) if indices_prot else slice(0, 0)
    new_cols_met = slice(all_indices.index(indices_met[0]), all_indices.index(indices_met[-1])+1) if indices_met else slice(0, 0)
    new_cols_clin_binary = slice(all_indices.index(indices_clin_bin[0]), all_indices.index(indices_clin_bin[-1])+1) if indices_clin_bin else slice(0, 0)
    new_cols_clin_cont = slice(all_indices.index(indices_clin_con[0]), all_indices.index(indices_clin_con[-1])+1) if indices_clin_con else slice(0, 0)
    cols_slice = slice(new_cols_prot.start, new_cols_met.stop)
    
    #### assign the filtered dataframe back to the dataframe df
    df = filtered_df

    ##################### Pre-process proteomics data ##############################################
    ##################################################################################################
    # Normalize each protein expression level relative to the median expression value for 
    ## the same protein across all patients (column-wise normalization)

    ###### This normalization step allows for adjusting for differences in baseline protein abundance:
    ###### some proteins are highly abundant, while others are present only in small amounts.
    ###### This normalization step will allow to identify group of proteins that follow similar patterns 
    ###### of regulation, regardless of differences between protein baseline abundances

    df.iloc[:, new_cols_prot] = df.iloc[:, new_cols_prot].apply(lambda x: x - x.median(), axis=0)
    
        
    ##################################################################################################    
    # Normalize protein expression values relative to the median expression value of
    ## all proteins for each patient. This operation will set the median expression value 
    ## of all proteins for each patient as equal to 0 (row-wise normalization)
    
    ###### This step normalizes protein samples loading, assuming equal loading for each sample
    
    row_medians_prot = df.iloc[:, new_cols_prot].median(axis=1)
    df.iloc[:, new_cols_prot] = df.iloc[:, new_cols_prot].sub(row_medians_prot, axis=0)
    
    print("Proteomics data pre-processing completed")
    
    ##################### Pre-process metabolomics data ##############################################
    ##################################################################################################
    # metabolites are expressed as fold change, log2 transform to equilibrate the distribution around 0
        
    df_met = df.iloc[:, new_cols_met]
    df_met = df_met.astype('float64')
    
    # Apply log2 transform to the slice of the DataFrame
    df.iloc[:, new_cols_met] = np.log2(df_met)
    

    # Normalize metabolite expression values relative to the median expression value for 
    ## the same metabolite across all patients (column-wise normalization)
    ###### This normalization step allows for adjusting for differences in baseline metabolite abundance:
    ###### some metabolites are highly abundant, while others are present only in small amounts.
    ###### This normalization step will allow to identify group of metabolites that follow similar patterns 
    ###### of regulation, regardless of differences between metabolite baseline abundances
    
    df.iloc[:, new_cols_met] = df.iloc[:, new_cols_met].apply(lambda x: x - x.median(), axis=0)


    # Normalize metabolite expression values relative to the median expression value of
    ## all metabolites for each patient. This operation will set the median expression value 
    ## of all metabolites for each patient as equal to 0 (row-wise normalization)
    ###### This step normalizes metabolite samples loading, assuming equal loading for each sample
    
    row_medians_met = df.iloc[:, new_cols_met].median(axis=1)
    df.iloc[:, new_cols_met] = df.iloc[:, new_cols_met].sub(row_medians_met, axis=0)

    print("Metabolomics data pre-processing completed")
    ##################################################################################################
    ####### Replace NaN values with column median values and zeros

    col_median = df.median(numeric_only=True)
    df = df.fillna(col_median)
    # df = df.fillna(0, axis=1) ## this line can be uncommented if any NaN values are left in the df after
    ## the previous operation and need to be filled out
    print("Missing values have been replaced")
    
    def remove_outliers(df, cols):
        pca = PCA(n_components=10)
        pca_values = pca.fit_transform(df.iloc[:, cols].values)
        cov = EmpiricalCovariance().fit(pca_values)
        dist = np.array([distance.mahalanobis(x, cov.location_, np.linalg.inv(cov.covariance_)) for x in pca_values])
        threshold = np.mean(dist) + np.std(dist)
        outliers = df.index[np.where(dist > threshold)[0]]
        outlier_labels = df.iloc[:, cols].index[outliers]
        df = df.drop(outlier_labels)
        # Reset the index and drop the old one.
        df = df.reset_index(drop=True)
        return df
    
    ##################################################################################################
    # Remove outlier patients for proteomics and metabolomics data
    orig_shape = df.shape[0]
    df = remove_outliers(df, new_cols_prot)
    filtered_shape = df.shape[0]
    samples_removed = orig_shape - filtered_shape
    print(f"{samples_removed} samples were removed after filtering outliers based on proteomics data")
    df = remove_outliers(df, new_cols_met)
    filtered_shape2 = df.shape[0]
    samples_removed2 = filtered_shape - filtered_shape2
    print(f"{samples_removed2} samples were removed after filtering outliers based on metabolomics data") 
    
    ##################### Pre-process clinical data ##############################################
    ##################################################################################################
    # Define a function to express continous clinical data as ratio values relative to the median value of 
    ## the same data
    def ratio_and_log_transform(df, cols_slice):
        for i in range(cols_slice.start, cols_slice.stop):
            # Calculate ratios by dividing by the column's median
            df.iloc[:, i] = df.iloc[:, i] / df.iloc[:, i].median()
            # Apply log2 transform, adding a small constant to avoid division by zero errors
            df.iloc[:, i] = np.log2(df.iloc[:, i] + 1e-9)
        return df
    
    # Define a function to scale the ratios so that they will have a distribution similar
    ## to that of the molecular expression data
    def scale_columns(df, cols_slice, min_val, max_val):
        for i in range(cols_slice.start, cols_slice.stop):
            # Apply min-max scaling to each column
            df.iloc[:, i] = (df.iloc[:, i] - df.iloc[:, i].min()) / (df.iloc[:, i].max() - df.iloc[:, i].min())
            # Rescale to desired range
            df.iloc[:, i] = df.iloc[:, i] * (max_val - min_val) + min_val
        return df
    
    # Compute the 0.1 and 99.9 percentiles of the molecular expression data values
    min_val = np.percentile(df.iloc[:, cols_slice].values.ravel(),  0.1)
    max_val = np.percentile(df.iloc[:, cols_slice].values.ravel(),  99.9)
    # Apply ratio calculation and scaling function to the continous clinical data
    df = ratio_and_log_transform(df, new_cols_clin_cont)
    df = scale_columns(df, new_cols_clin_cont, min_val, max_val)
    
    # Standardize binary clinical data (optimized for sex) to set the values as equal to min_val and
    ## max_val, calculated above
    df.iloc[:, new_cols_clin_binary] = df.iloc[:, new_cols_clin_binary].replace({0.0: min_val, 1.0: max_val})
    print("Clinical data pre-processing completed")
    return(df)


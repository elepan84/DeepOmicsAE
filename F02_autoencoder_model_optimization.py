#!/usr/bin/env python
# coding: utf-8

# In[46]:


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
RANDOM_STATE = 55 ## We will pass it to every sklearn call so we ensure reproducibility


# In[47]:


#### This function performs feature selection from a given DataFrame.
#     The function uses the SelectKBest method from the sklearn library to select
#     the 'k_feat' best features based on the f_classif score.

#     Parameters:
#     df (DataFrame): The input DataFrame. 
#     k_feat (int): The number of top features to select.
#     cols_X (list): The column indices for the features (X).
#     y_label (string): The column name for the target variable (y).

#     Returns:
#     X_new (array): The values of the selected features.
#     df_selected (DataFrame): DataFrame of the selected features, preserving original indices.
    

def feature_selection(df, k_feat, cols_X, y_label):   
    X = df.iloc[:, cols_X]
    y = df[y_label]
    #import pdb; pdb.set_trace()
    # feature selection
    selector = SelectKBest(f_classif, k=k_feat)  
    X_new = selector.fit_transform(X, y)

    # Get the selected features and their indices
    selected_indices = selector.get_support(indices=True)
    selected_columns = X.columns[selected_indices]

    # Create df_selected with selected features and preserved index values
    df_selected = pd.DataFrame(X_new, columns=selected_columns, index=df.index)
    return(X_new, df_selected)


# In[48]:


def make_encoder(n_feat, n2, n3, latent):
    ### This function constructs the encoder part of the autoencoder model. 
    ### It takes as input the number of features, and the number of neurons for the three layers.
    # The model architecture consists of two hidden layers followed by a latent space representation.

    # Parameters:
    # n_feat (int): The number of features in the input data.
    # n2 (int): The number of neurons in the first hidden layer.
    # n3 (int): The number of neurons in the second hidden layer.
    # latent (int): The number of neurons in the latent space representation.

    # Returns:
    # keras.Model: The constructed encoder model.
    
    inputs = keras.Input(shape=(n_feat,))
    x = layers.Dense(n2, activation='relu')(inputs)
    x = layers.Dense(n3, activation='relu')(x)
    z_mean = layers.Dense(latent)(x)
    z_log_var = layers.Dense(latent)(x)
    return keras.Model(inputs, [z_mean, z_log_var], name='encoder')



# In[49]:


# Define the decoder
def make_decoder(n_feat, n2, n3, latent):
    ### This function constructs the decoder part of the autoencoder model. 
    ### It takes as input the number of features, and the number of neurons for the three layers.
    # The model architecture consists of two hidden layers followed by an output layer with
    # the same size as the input data.

    # Parameters:
    # n_feat (int): The number of features in the input data.
    # n2 (int): The number of neurons in the first hidden layer.
    # n3 (int): The number of neurons in the second hidden layer.
    # latent (int): The number of neurons in the latent space representation.

    # Returns:
    # keras.Model: The constructed decoder model.
        
    latent_inputs = keras.Input(shape=(latent,))
    x = layers.Dense(n3, activation='relu')(latent_inputs)
    x = layers.Dense(n2, activation='relu')(x)
    outputs = layers.Dense(n_feat)(x)
    return keras.Model(latent_inputs, outputs, name='decoder')



# In[50]:


# Define the sampling layer
def create_sampling_layer():
    ### This function creates and returns an instance of a custom Sampling layer. 
    ### This layer uses (z_mean, z_log_var) to sample a point 'z' from the latent distribution.

    # Returns:
    # Sampling: An instance of the custom Sampling layer.

    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z"""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    return Sampling() # <-- returns an instance of the Sampling class


# In[51]:


# Define the full AE model by combining the encoder, decoder, and sampling layers
def ae_model(n_feat, n2, n3, latent):
    ### This function creates the full autoencoder model by combining the encoder, decoder, 
    ### and sampling layers. It takes as input the number of features, and the number of neurons 
    ### for the three layers.
    inputs = keras.Input(shape=(n_feat,))
    encoder = make_encoder(n_feat, n2, n3, latent)
    decoder = make_decoder(n_feat, n2, n3, latent)
    #sampling = Sampling()
    sampling = create_sampling_layer()
    z_mean, z_log_var = encoder(inputs)
    z = sampling([z_mean, z_log_var])
    outputs = decoder(z)
    ae = keras.Model(inputs, outputs, name='ae')
    return(ae, encoder)


# In[52]:


### run it first with n_epochs set to 300, then use min_val_loss_epochs for the final run
## verbose can be either set to 1 to visualize the losses as the model runs, or 0 to silence them
def ae_model_run(ae, X, X_train, X_val, n_epochs, verbose, encoder):
    ### This function compiles and trains the autoencoder model, then applies it to the input data X.
    ### It also calculates and returns the validation loss history and the minimum validation loss epoch.

    # Parameters:
    # ae (keras.Model): The autoencoder model to train and apply to the data.
    # X (DataFrame): The input data on which the model will be applied after training.
    # X_train (DataFrame): The training data.
    # X_val (DataFrame): The validation data used during training.
    # n_epochs (int): The number of epochs to train the model.
    # verbose (int): Determines whether to display loss information during training (1 = yes, 0 = no).
    # encoder (keras.Model): The encoder part of the autoencoder model.

    # Returns:
    # history (History): The training history, including loss and validation loss values at each epoch.
    # val_loss (list): A list of validation loss values at each epoch.
    # min_val_loss_epoch (int): The epoch at which the minimum validation loss was achieved.
    # bottleneck_features (ndarray): The encoded features produced by applying the trained encoder to the input data.
    
    ae.compile(optimizer='adam', loss=losses.MeanSquaredError())

    history = ae.fit(X_train, X_train,
    epochs=n_epochs,
    validation_data=(X_val, X_val), verbose=verbose)

    encoder_output = encoder.predict(X)
    bottleneck_features = encoder_output[0]
    val_loss = history.history['val_loss']
    min_val_loss = min(val_loss)
    min_val_loss_epoch = val_loss.index(min_val_loss) + 1
    return (history, val_loss, min_val_loss_epoch, bottleneck_features, encoder)


# In[53]:


def ae_model_setup_and_run(n_feat, n2, n3, latent, X, X_train, X_val, n_epochs, verbose):
    # This function sets up the autoencoder model, trains it, applies it to the data, and returns various outputs.

    # Parameters:
    # n_feat (int): The number of features in the input data.
    # n2 (int): The number of neurons in the first hidden layer.
    # n3 (int): The number of neurons in the second hidden layer.
    # latent (int): The number of neurons in the latent space representation.
    # X (DataFrame): The input data on which the model will be applied after training.
    # X_train (DataFrame): The training data.
    # X_val (DataFrame): The validation data used during training.
    # n_epochs (int): The number of epochs to train the model.
    # verbose (int): Determines whether to display loss information during training (1 = yes, 0 = no).

    # Returns:
    # history (History): The training history, including loss and validation loss values at each epoch.
    # val_loss (list): A list of validation loss values at each epoch.
    # min_val_loss_epoch (int): The epoch at which the minimum validation loss was achieved.
    # bottleneck_features (ndarray): The encoded features produced by applying the trained encoder to 
    # the input data.
    
    ae, encoder = ae_model(n_feat, n2, n3, latent)
    history, val_loss, min_val_loss_epoch, bottleneck_features,encoder = ae_model_run(ae, X, X_train, X_val, n_epochs, verbose, encoder)
    if min_val_loss_epoch < 20: min_val_loss_epoch = 20
    return history, val_loss, min_val_loss_epoch, bottleneck_features, encoder



# In[54]:


def combinations(df, cols_X_prot, cols_X_met, n_comb):
    # Define the parameter values
    no_samples = df.shape[0]
    prot_cols = cols_X_prot.stop - cols_X_prot.start
    met_cols = cols_X_met.stop - cols_X_met.start
    
    # Define the parameter values
    min_val = 50
    max_val_prot = round(prot_cols * 0.35)
    max_val_met = round(met_cols * 0.35)
    latent_min = round(no_samples * 0.01)
    if latent_min == 1 or latent_min == 2: latent_min = 3
    latent_max = round(no_samples * 0.025) + 1
    if latent_max == 1 or latent_max == 2: latent_max = 3
    latent_n = latent_max - latent_min
    if latent_n == 0: latent_n = 1
    kprot_values = sorted(random.sample(range(min_val, max_val_prot), 150))  # Randomly select 150 values from the range
    kmet_values = sorted(random.sample(range(min_val, max_val_met), 75))  # Randomly select 75 values from the range
    latent_values = random.sample(range(latent_min, latent_max), latent_n)  # Randomly select latent_n values from the range
    
    # Define the percentiles to select
    percentiles1 = [0, 2.5, 5, 7.5, 10]
    percentiles2 = [0, 4.5, 9, 13.5, 18]
    # Define the range values
    range_prot = max_val_prot - min_val
    range_met = max_val_met - min_val

    # Select values at these percentiles for kprot and kmet
    selected_kprot_values_bottom = [round(min_val + np.percentile(range(range_prot), percentile)) for percentile in percentiles1]
    selected_kmet_values_bottom = [round(min_val + np.percentile(range(range_met), percentile)) for percentile in percentiles2]

    # Select values at these percentiles for kprot and kmet
    selected_kprot_values_top = [round(min_val + np.percentile(range(range_prot), 100 - percentile)) for percentile in percentiles1]
    selected_kmet_values_top = [round(min_val + np.percentile(range(range_met), 100 - percentile)) for percentile in percentiles2]    
    
    # Shuffle the kprot and kmet lists
    random.shuffle(selected_kprot_values_bottom)
    random.shuffle(selected_kmet_values_bottom)
    random.shuffle(selected_kprot_values_top)
    random.shuffle(selected_kmet_values_top)

    # Select one element from each list to form a combination
    selected_bottom_combinations = [(kprot, kmet, random.choice(latent_values)) for kprot, kmet in zip(selected_kprot_values_bottom[:5], selected_kmet_values_bottom[:5])]
    selected_top_combinations = [(kprot, kmet, random.choice(latent_values)) for kprot, kmet in zip(selected_kprot_values_top[:5], selected_kmet_values_top[:5])]

    # Generate all parameter combinations
    all_combinations = []
    for kprot in kprot_values:
        for kmet in kmet_values:
            for latent in latent_values:
                all_combinations.append((kprot, kmet, latent))

    # Subtract the selected combinations from all combinations
    remaining_combinations = [comb for comb in all_combinations if comb not in selected_bottom_combinations and comb not in selected_top_combinations]

    # Randomly select from the remaining combinations
    random.shuffle(remaining_combinations)
    selected_random_combinations = remaining_combinations[:(n_comb - 10)]  # Exclude 10 combinations that were selected based on percentiles

    # Combine the selected combinations
    selected_combinations = selected_bottom_combinations + selected_top_combinations + selected_random_combinations
    return(selected_combinations)


# In[55]:


def model_optimization(df, n_all_feat, n_comb, cols_X_prot, 
                       cols_X_met, cols_clin, cols_X_expr, y_label):
    
    # This function performs model optimization by randomly sampling possible combinations of parameters 
    # and running feature selection, model training, and evaluation for each combination.

    # Parameters:
    # df: The input DataFrame containing the features and target variable
    # n_all_feat: Total number of features in the dataframe
    # n_comb: Number of combinations of parameters to test
    # cols_X_prot: Columns in the dataframe that represent proteomics data
    # cols_X_met: Columns in the dataframe that represent metabolomics data
    # cols_clin: Columns in the dataframe that represent clinical data
    # cols_X_expr: Columns in the dataframe that represent the input features for PCA
    # y_label: The column in the dataframe that represents the target variable

    # Returns:
    # df_results: A DataFrame containing the results of the model optimization
    # df_selected: The final selected features in a DataFrame

    
    # Initialize an empty DataFrame with columns
    columns = ['kprot', 'kmet', 'latent', 'sil_score_initial',
              'sil_score_final', 'pca_all', 'pca_extracted']

    df_results = pd.DataFrame(columns=columns)
    selected_combinations = combinations(df, cols_X_prot,cols_X_met, n_comb)

    # Perform PCA
    X = df.iloc[:, cols_X_expr]
    y = df[y_label]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    silhouette_feat = silhouette_score(X, y)
    
    # Calculate Silhouette Scores
    pca_silhouette_score_all_feat = silhouette_score(X_pca, y)
    
    i = 0
    # Loop over each combination
    for kprot, kmet, latent in selected_combinations:
        i = i +1
        print(i)
        # Select features
        print("Selecting features")
        X_new_prot, df_selected_prot = feature_selection(df, kprot, cols_X_prot, y_label)
        X_new_met, df_selected_met = feature_selection(df, kmet, cols_X_met, y_label)
        df_selected = pd.concat([df_selected_prot, 
                                 df_selected_met, df.iloc[:, cols_clin], df[y_label]], axis=1)

        # Define the autoencoder model architecture
        n_feat = df_selected.iloc[:, :-1].shape[1]
        n2 = round(n_feat/1.5)
        if (n2 <= 70):
            n3 = round(n2/2)
        if (n2 > 70) and (n2 <= 200):
            n3 = round(n2/5)
        else:
            n3 = round(n2/6)
        
        # Separate features and labels
        X = df_selected.iloc[:, :-1]
        y = df_selected.iloc[:, -1]

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                          test_size=0.2, random_state=42)

        print("Extracting features")
        ### run the autoencoder first with n_epochs set to 300, then use min_val_loss_epochs for the final run
        ## verbose can be either set to 1 to visualize the losses as the model runs, or 0 to silence them

        history, val_loss, min_val_loss_epoch, bottleneck_features, encoder = ae_model_setup_and_run(n_feat, n2, n3, latent, X, X_train, X_val, 300, verbose = 0)
        history, val_loss, min_val_loss_epoch, bottleneck_features, encoder = ae_model_setup_and_run(n_feat, n2, n3, latent, X, X_train, X_val, min_val_loss_epoch, verbose = 0)                                                                                       

        # Create a pandas DataFrame for the extracted bottleneck features
        extracted_features_df = pd.DataFrame(bottleneck_features, columns=[f"Feature_{i}" for i in range(latent)])
        extracted_features_df[y_label] = y

        ##### perform PCA and tSNE again after selecting features

        # Separate features and labels
        X = bottleneck_features

        # Perform PCA and t-SNE
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Calculate Silhouette Scores
        pca_silhouette_score_extracted_feat = silhouette_score(X_pca, y)
        silhouette_bottleneck = silhouette_score(bottleneck_features, y)


        data = {
        'kprot': [kprot],
        'kmet': [kmet],
        'latent': [latent],
        'sil_score_initial': [silhouette_feat],
        'sil_score_final': [silhouette_bottleneck],
        'pca_all': [pca_silhouette_score_all_feat],
        'pca_extracted': [pca_silhouette_score_extracted_feat],
        }

        new_row = pd.DataFrame(data)

        ## Add the results to the results DataFrame
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        
    return(df_results, df_selected)


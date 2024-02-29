#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:04:59 2023

@author: juanpablomayaarteaga
"""

import pandas as pd
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from morphoglia import set_path
import hdbscan
import sklearn.datasets
import pandas as pd
import numpy as np
import umap
import umap.plot
import time
import os

# Record start time
start_time = time.time()


# Load the data
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Plot_path = set_path(o_path + "Plots/")
UMAP_path = set_path(o_path + "Plots/UMAP/")


data = pd.read_csv(csv_path + "Morphology.csv")


######################################################################
######################################################################
######################################################################
######################################################################



features_to_assess =  ['Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 
                    'Junctions', 'Initial_Points', 'ratio_branches', 
                    'Convex_Hull_eccentricity', 
                    'Cell_compactness', 'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Cell_solidity', 'Cell_convexity',
                    'Sholl_circles']



# Extract the selected features from the dataset
selected_data = data[features_to_assess]




######################################################################
######################################################################
######################################################################
######################################################################

#############################   UMAP   ###############################


######################################################################
######################################################################
######################################################################
######################################################################



# Apply UMAP for dimensionality reduction using the PCA result
n_neighbors = 15
min_dist= 0.01
min_cluster_size= 20
min_samples=20

reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=24)
embedding = reducer.fit_transform(selected_data)
#embedding = reducer.fit_transform(selected_pca_result)

# Visualize the data with UMAP
plt.style.use('dark_background')
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
plt.title(f'UMAP (n={n_neighbors}, d={min_dist})', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.grid(False)
plt.style.use('dark_background')

plt.savefig(UMAP_path + f"UMAP_{n_neighbors}_{min_dist}.png", dpi=500)
plt.show()  # Show the plot

# Save the updated dataframe to a new CSV file
# data.to_csv(csv_path + "Morphology_UMAP.csv", index=False)





######################################################################
######################################################################
######################################################################
######################################################################

############################   HDBSCAN    #############################


######################################################################
######################################################################
######################################################################
######################################################################


# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=24)
embedding = reducer.fit_transform(selected_data)

# Apply HDBSCAN clustering to the UMAP-transformed data
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, allow_single_cluster=True)
clusterer.fit(embedding)
labels = clusterer.fit_predict(embedding)

# Define a mapping dictionary for label replacement
label_mapping = {
    0: 1,
    1: 0,
    4: 5,
    5: 4
}

# Define the cluster colors RGB
cluster_colors = {
    0: "crimson",         
    1: "orangered",       
    4: "mediumturquoise", 
    5: "paleturquoise",   
    2: "gold",
    3: "limegreen",
    -1: "darkorchid"
}

# Convert cluster labels to colors with modified labels
default_color = "gray"
modified_labels = np.array([label_mapping.get(label, label) for label in labels])
cluster_colors_array = np.array([cluster_colors.get(label, default_color) for label in modified_labels])

# Visualize the clustered data with colors
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array, alpha=0.3)

plt.title(f'HDBSCAN Clustering', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.grid(False)
plt.style.use('dark_background')

# Add cluster labels to the plot with modified labels
for cluster_label in np.unique(modified_labels):
    if cluster_label != -1:  # Excluding noise points labeled as -1
        cluster_points = embedding[modified_labels == cluster_label]
        cluster_center = np.mean(cluster_points, axis=0)
        plt.text(cluster_center[0], cluster_center[1], str(cluster_label), fontsize=10, color='white')

plt.savefig(UMAP_path + f"UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.png", dpi=500)
plt.show()


# Add a column to the original dataframe with cluster labels
data['UMAP_1'] = embedding[:, 0]
data['UMAP_2'] = embedding[:, 1]
data['Cluster_Labels'] = clusterer.labels_

# Apply the label mapping to update the cluster labels in the dataframe
data['Cluster_Labels'] = data['Cluster_Labels'].map(label_mapping).fillna(data['Cluster_Labels'])

# Save the updated dataframe to a new CSV file
data.to_csv(csv_path + f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv", index=False)


"""

######### INTERACTIVE
p = umap.plot.interactive(
    reducer,
    labels=data['Cluster_Labels'],
    hover_data=data[['Cell_ID']],
    color_key=cluster_colors,
    point_size=8
)
umap.plot.show(p)

"""
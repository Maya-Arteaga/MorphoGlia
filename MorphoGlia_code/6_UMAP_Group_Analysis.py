#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 02:42:53 2024

@author: juanpablomaya
"""

import pandas as pd
import umap
import matplotlib.pyplot as plt
import numpy as np
from morphoglia import set_path
import hdbscan
import time
import os

# Record start time
start_time = time.time()

# Load the data
i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Data/")
Plot_path = set_path(o_path + "Plots/")
UMAP_path = set_path(o_path + "Plots/UMAP_Diagnosis/")

n_neighbors = 10
min_dist = 0.1
min_cluster_size = 20
min_samples = 10

data = pd.read_csv(csv_path + f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv")

# Define a mapping of old category names to new names
category_mapping = {
    "VEH_SS_CA1": "SS-CA1",
    "VEH_SCO_CA1": "SCOP-CA1",
    "VEH_SS_HILUS": "SS-Hilus",
    "VEH_SCO_HILUS": "SCOP-Hilus"
}

# Replace old category names with new names in the data
data['categories'] = data['categories'].replace(category_mapping)

features_to_assess = ['Soma_area', 'Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio',
                      'End_Points', 'Branches', 'Convex_Hull_eccentricity',
                      'Cell_area', 'Cell_compactness', 'Cell_feret_diameter', 'Cell_eccentricity',
                      'Cell_aspect_ratio', 'Sholl_max_distance', 'Cell_solidity', 'Cell_convexity']

# Extract the selected features from the dataset
selected_data = data[features_to_assess]

# Convert Cluster_Labels column to string type
data['Clusters'] = data['Clusters'].astype(int)

# Apply UMAP for dimensionality reduction
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=24)
embedding = reducer.fit_transform(selected_data)

# Apply HDBSCAN clustering to the UMAP-transformed data
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, allow_single_cluster=True)
clusterer.fit(embedding)
labels = clusterer.fit_predict(embedding)

# Define a mapping dictionary for label replacement
label_mapping = {
    0: 0,
    1: 4,
    2: 3,
    3: 2,
    4: 1,
    5: 5,
    6: 6
}

# Apply the label mapping
modified_labels = np.array([label_mapping.get(label, label) for label in labels])

# Define the cluster colors
cluster_colors = {
    0: "darkorchid",
    1: "crimson",
    2: "gold",
    3: "limegreen",
    4: "mediumturquoise",
    5: "paleturquoise",
    -1: "white"
}

# Convert modified cluster labels to colors
default_color = "gray"
cluster_colors_array = np.array([cluster_colors.get(label, default_color) for label in modified_labels])

# Add UMAP embedding and modified cluster labels to the data
data['UMAP_1'] = embedding[:, 0]
data['UMAP_2'] = embedding[:, 1]
data['Clusters'] = modified_labels

# Define the colors for each category
category_colors = {
    "SS-CA1": "mediumturquoise",
    "SCOP-CA1": "darkorchid",
    "SS-Hilus": "limegreen",
    "SCOP-Hilus": "crimson"
}




# Visualize HDBSCAN clustering with colored groups
for category, color in category_colors.items():
    plt.figure()
    
    # Plot all categories in light gray as background
    other_categories = data['categories'] != category
    plt.scatter(
        embedding[other_categories, 0],
        embedding[other_categories, 1],
        c='lightgray',
        alpha=0.8
    )
    
    # Plot the selected category with HDBSCAN cluster colors
    category_data = data['categories'] == category
    plt.scatter(
        embedding[category_data, 0],
        embedding[category_data, 1],
        c=cluster_colors_array[category_data],
        alpha=0.3
    )

    plt.title(f'{category}', fontsize=18)
    plt.xlabel('UMAP 1', fontsize=14)
    plt.ylabel('UMAP 2', fontsize=14)
    plt.grid(False)
    plt.savefig(UMAP_path + f"UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}_Category_{category}.png", dpi=1200)
    plt.show()



######################################################################
########################    NOISE   ##################################
######################################################################

# Define the cluster colors RGB
cluster_colors = {
    -1: "darkorange",
    0: "gainsboro",
    1: "gainsboro",
    3: "gainsboro",
    2: "gainsboro",
    4: "gainsboro",
    5: "gainsboro",
    6: "gainsboro"
}

# Convert cluster labels to colors
default_color = "gray"
cluster_colors_array = np.array([cluster_colors.get(label, default_color) for label in clusterer.labels_])

# Visualize the clustered data with colors
#plt.figure(figsize=(12, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array, alpha=0.5)
plt.title(f'HDBSCAN: Noise (n_neighbors ={n_neighbors}, min_dist={min_dist})', fontsize=14)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.grid(False)

# Add cluster labels to the plot
for cluster_label in np.unique(clusterer.labels_):
    if cluster_label != -1:  # Excluding noise points labeled as -1
        cluster_points = embedding[clusterer.labels_ == cluster_label]
        cluster_center = np.mean(cluster_points, axis=0)
        plt.text(cluster_center[0], cluster_center[1], str(cluster_label), fontsize=10, color='white')

plt.savefig(UMAP_path + f"UMAP_HDBSCAN_{n_neighbors}_{min_dist}_noise.png", dpi=900)
plt.show()




# Record end time
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_min = elapsed_time / 60

print("")
print(f"Elapsed Time: {elapsed_time} seconds")
print(f"Elapsed Time: {elapsed_time_min} min")
print("")
print("________________________________________")
print("________________________________________")

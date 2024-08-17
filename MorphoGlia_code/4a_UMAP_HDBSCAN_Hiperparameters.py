#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:15:48 2023

@author: juanpablomayaarteaga
"""

#pip install umap-learn
#pip install hdbscan


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

# Record start time
start_time = time.time()

method="Random_Forest"

#From 3_Correlation_RFE.py
test=0.3
#test=test_size

# Load the data
i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
UMAP_path = set_path(o_path + f"Plots/UMAP_trials_{method}_{test}/")
HDBSCAN_path = set_path(o_path + f"Plots/HDBSCAN_trials_{method}_{test}/")
data = pd.read_csv(csv_path + "Morphology.csv")


######################################################################
######################################################################
######################################################################
######################################################################

#print(selected_features)

#Test 0.4

features_to_assess= ['Soma_area', 'Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 
                     'End_Points', 'Branches', 
                     'Convex_Hull_eccentricity', 
                     'Cell_area', 'Cell_compactness', 'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 
                     'Sholl_max_distance', 'Cell_solidity', 'Cell_convexity']

#From 3_Correlation_RFE.py
#features_to_assess=selected_features




# Extract the selected features from the dataset
selected_data = data[features_to_assess]


# Define parameters
n_neighbors = [10, 15, 20, 30, 50]
min_dist = [0.01, 0.05, 0.1]

# Create a figure with 5 rows and 3 columns
fig, axes = plt.subplots(nrows=len(n_neighbors), ncols=len(min_dist), figsize=(22, 25))

# Loop through each combination of n_neighbors and min_dist
for i, n in enumerate(n_neighbors):
    for j, d in enumerate(min_dist):
        # Apply UMAP
        reducer = umap.UMAP(n_neighbors=n, min_dist=d, random_state=24)
        embedding = reducer.fit_transform(selected_data)

        # Plot the UMAP result
        ax = axes[i, j]
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.4)
        ax.set_title(f'n_neighbors={n}, min_dist={d}', fontsize=14)
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.grid(False)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(UMAP_path + f"UMAP_grid_{test}.png", dpi=800)
plt.show()


# Apply UMAP for dimensionality reduction using the PCA result
n_neighbors = [10, 15, 20, 30, 50]
min_dist= [0.01, 0.05, 0.1]


for n in n_neighbors:
    for d in min_dist:
        
        reducer = umap.UMAP(n_neighbors=n, min_dist=d, random_state=24)
        embedding = reducer.fit_transform(selected_data)

        
        # Visualize the data with UMAP
        #plt.style.use('dark_background')
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
        plt.title(f'UMAP (n={n} d={d} )', fontsize=18)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.grid(False)
        #plt.style.use('dark_background')
        
        plt.savefig(UMAP_path + f"UMAP_{n}_{d}_{test}.png", dpi=1200)
        plt.show()  # Show the plot

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
n_neighbors = [5, 10, 15, 20, 30, 50, 100]
#min_dist= [0.01, 0.05, 0.1]
min_dist= [0.1]

for n in n_neighbors:
    for d in min_dist:
        
        reducer = umap.UMAP(n_neighbors=n, min_dist=d, random_state=24)
        embedding = reducer.fit_transform(selected_data)

        
        # Visualize the data with UMAP
        #plt.style.use('dark_background')
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
        plt.title(f'UMAP (n={n} d={d} )', fontsize=18)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.grid(False)
        #plt.style.use('dark_background')
        
        plt.savefig(UMAP_path + f"UMAP_{n}_{d}_{test}.png", dpi=300)
        plt.show()  # Show the plot


######################################################################
######################################################################
######################################################################
######################################################################

############################   HDBSCAN    #############################


######################################################################
######################################################################
######################################################################
######################################################################
n_neighbors = [5, 10, 15]
min_dist= [0.01, 0.05, 0.1]
min_cluster_size=[10, 15, 20] 
min_samples=[10, 15, 20]




for n in n_neighbors:
    for d in min_dist: 
        for c in min_cluster_size:
            for s in min_samples:
                
                reducer = umap.UMAP(n_neighbors=n, min_dist=d, random_state=24)
                embedding = reducer.fit_transform(selected_data)
                # Apply HDBSCAN clustering to the UMAP-transformed data
                clusterer = hdbscan.HDBSCAN(min_cluster_size=c, min_samples=s, allow_single_cluster=True)
                clusterer.fit(embedding)
                labels = clusterer.fit_predict(embedding)
                
                # Define the cluster colors RGB
                # Custom color palette
                cluster_colors = {
                
                    0: "darkorchid",         
                    1: "crimson",       
                    2: "orangered", 
                    3: "gold",   
                    4: "limegreen",
                    5: "mediumturquoise",
                    6: "paleturquoise", 
                    -1: "white"
                }
                                
                # Convert cluster labels to colors
                #cluster_colors_array = np.array([cluster_colors.get(label, (0, 0, 0)) for label in clusterer.labels_])
                default_color = "gray"
                cluster_colors_array = np.array([cluster_colors.get(label, default_color) for label in clusterer.labels_])
                
                
                # Visualize the clustered data with colors
                #plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array / 255.0, alpha=0.5)
                plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array, alpha=0.3)
                
                
                
                plt.title(f'HDBSCAN  (n={n} d={d} c={c} s={s})', fontsize=18)
                plt.xlabel('UMAP 1', fontsize=14 )
                plt.ylabel('UMAP 2', fontsize=14 )
                plt.grid(False)
                #plt.style.use('dark_background')
                
                # Add cluster labels to the plot
                for cluster_label in np.unique(clusterer.labels_):
                    if cluster_label != -1:  # Excluding noise points labeled as -1
                        cluster_points = embedding[clusterer.labels_ == cluster_label]
                        cluster_center = np.mean(cluster_points, axis=0)
                        plt.text(cluster_center[0], cluster_center[1], str(cluster_label), fontsize=10, color='white')
                
                plt.savefig(HDBSCAN_path + f"UMAP_HDBSCAN_{n}_{d}_{c}_{s}_{test}.png", dpi=300)
                plt.show()  




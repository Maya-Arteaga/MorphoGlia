#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:15:48 2023

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

# Record start time
start_time = time.time()

test=0.4

# Load the data
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
UMAP_path = set_path(o_path + f"Plots/UMAP_trials_test_{test}/")
HDBSCAN_path = set_path(o_path + f"Plots/HDBSCAN_trials_test_{test}/")
data = pd.read_csv(csv_path + "Morphology.csv")


######################################################################
######################################################################
######################################################################
######################################################################


features_to_assess = ['Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 
                      'Junctions', 'Initial_Points', 'ratio_branches', 
                      'Convex_Hull_eccentricity', 
                      'Cell_compactness', 'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Cell_solidity', 'Cell_convexity',
                      'Sholl_circles',]






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
n_neighbors = [5, 10, 15, 20, 30, 50, 100]
min_dist= [0.01, 0.05, 0.1]

for n in n_neighbors:
    for d in min_dist:
        
        reducer = umap.UMAP(n_neighbors=n, min_dist=d, random_state=24)
        embedding = reducer.fit_transform(selected_data)

        
        # Visualize the data with UMAP
        plt.style.use('dark_background')
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
        plt.title(f'UMAP (n={n} d={d} )', fontsize=18)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.grid(False)
        plt.style.use('dark_background')
        
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
                    -1: "darkorchid",
                    2: "orangered",
                    3: "crimson",
                    0: "paleturquoise",
                    4: "gold",
                    5: "limegreen",
                    1: "mediumturquoise",
                    #6: "lime" darkorchid
                    #seagreen darkorchid
                
                    
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
                plt.style.use('dark_background')
                
                # Add cluster labels to the plot
                for cluster_label in np.unique(clusterer.labels_):
                    if cluster_label != -1:  # Excluding noise points labeled as -1
                        cluster_points = embedding[clusterer.labels_ == cluster_label]
                        cluster_center = np.mean(cluster_points, axis=0)
                        plt.text(cluster_center[0], cluster_center[1], str(cluster_label), fontsize=10, color='white')
                
                plt.savefig(HDBSCAN_path + f"UMAP_HDBSCAN_{n}_{d}_{c}_{s}_{test}.png", dpi=300)
                plt.show()  




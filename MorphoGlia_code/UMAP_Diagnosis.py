#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 00:23:24 2024

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

i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Plot_path = set_path(o_path + "Plots/")
UMAP_path = set_path(o_path + "Plots/UMAP_Diagnosis/")


n_neighbors = 10
min_dist= 0.1
min_cluster_size= 20
min_samples=10


data = pd.read_csv(csv_path + f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv")

"""
# Filter out rows where Cluster_Labels column has a value of -1
data = data[data['Cluster_Labels'] != -1]

"""


#test 0.3
features_to_assess=  ['Soma_area', 'Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 
                      'End_Points', 'Branches', 'Convex_Hull_eccentricity', 
                      'Cell_area', 'Cell_compactness', 'Cell_feret_diameter', 'Cell_eccentricity', 
                      'Cell_aspect_ratio', 'Sholl_max_distance', 'Cell_solidity', 'Cell_convexity']


"""
features_to_assess =  ['Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 
                    'Junctions', 'Initial_Points', 'ratio_branches', 
                    'Convex_Hull_eccentricity', 
                    'Cell_compactness', 'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Cell_solidity', 'Cell_convexity',
                    'Sholl_circles']


"""
# Extract the selected features from the dataset
selected_data = data[features_to_assess]




# Convert Cluster_Labels column to string type
data['Clusters'] = data['Clusters'].astype(int)


reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=24)
embedding = reducer.fit_transform(selected_data)
#embedding = reducer.fit_transform(selected_pca_result)

# Visualize the data with UMAP
#plt.style.use('dark_background')
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
plt.title(f'UMAP (n={n_neighbors}, d={min_dist})', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.grid(False)
#plt.style.use('dark_background')

#plt.savefig(UMAP_path + f"UMAP_{n_neighbors}_{min_dist}.png", dpi=500)
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

#0.5,0.1

label_mapping = {
    0: 0,
    1: 4,
    2: 3,
    3: 2,
    4: 1,
    5: 5,
    6: 6
}


# Define the cluster colors RGB

cluster_colors = {

    0: "darkorchid",         
    1: "crimson",       
    #2: "orangered", 
    2: "gold",   
    3: "limegreen",
    4: "mediumturquoise",
    6: "paleturquoise", 
    -1: "white"
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
#plt.style.use('dark_background')

# Add cluster labels to the plot with modified labels
for cluster_label in np.unique(modified_labels):
    if cluster_label != -1:  # Excluding noise points labeled as -1
        cluster_points = embedding[modified_labels == cluster_label]
        cluster_center = np.mean(cluster_points, axis=0)
        plt.text(cluster_center[0], cluster_center[1], str(cluster_label), fontsize=10, color='black')

plt.savefig(UMAP_path + f"UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.png", dpi=900)
plt.show()


# Add a column to the original dataframe with cluster labels
data['UMAP_1'] = embedding[:, 0]
data['UMAP_2'] = embedding[:, 1]
data['Clusters'] = clusterer.labels_


######################################################################
######################################################################
########################    NOISE   ##################################
######################################################################
######################################################################


# Define the cluster colors RGB
# Custom color palette
cluster_colors = {
    -1: "darkorchid",
    0: "gainsboro",
    1: "gainsboro",
    3: "gainsboro",
    2: "gainsboro",
    4: "gainsboro",
    5: "gainsboro",
    6: "gainsboro"

    
}

# Convert cluster labels to colors
#cluster_colors_array = np.array([cluster_colors.get(label, (0, 0, 0)) for label in clusterer.labels_])
default_color = "gray"
cluster_colors_array = np.array([cluster_colors.get(label, default_color) for label in clusterer.labels_])


# Visualize the clustered data with colors
#plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array / 255.0, alpha=0.5)
plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_colors_array, alpha=0.5)



plt.title(f'HDBSCAN: NOISE (n ={n_neighbors})', fontsize=18)
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

plt.savefig(UMAP_path + f"UMAP_HDBSCAN_{n_neighbors}_{min_dist}_noise.png", dpi=500)
plt.show()  # Show the plot




######################################################################
######################################################################
######################################################################
######################################################################

#######################   SEPARATED BY GROUP   #######################


######################################################################
######################################################################
######################################################################
######################################################################








import matplotlib.pyplot as plt
import umap

# Assuming data is your DataFrame, and embedding is the result of UMAP
#n_neighbors = 15
reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=24)
embedding = reducer.fit_transform(selected_data)  

cluster_colors = {
    "VEH_SS_CA1": "mediumturquoise",
    "VEH_SCO_CA1": "darkorchid",
    "VEH_SS_HILUS": "limegreen",
    "VEH_SCO_HILUS": "crimson"
    
    
}

# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
#plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=0.5,
        label=category
    )

plt.title(f'UMAP Analysis by Group (n_neighbors={n_neighbors})', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(UMAP_path + f"UMAP_{n_neighbors}_{min_dist}_Categories.png", dpi=800)
plt.show()




######################################################################
######################################################################
#######################      VEH_SS_CA1     ##############################
######################################################################
######################################################################

cluster_colors = {
    "VEH_SS_CA1": "mediumturquoise",
    "VEH_SCO_CA1": "gainsboro",
    "VEH_SS_HILUS": "gainsboro",
    "VEH_SCO_HILUS": "gainsboro"
    
    
}



# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
#plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=1.0,
        label=category
    )

plt.title('UMAP Analysis by Group: VEH SS', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(UMAP_path + f"UMAP_{n_neighbors}_{min_dist}_Categories_VEH_SS.png", dpi=800)
plt.show()



######################################################################
######################################################################
#######################      VEH_SCO_CA1     ##############################
######################################################################
######################################################################

cluster_colors = {
    "VEH_SS_CA1": "gainsboro",
    "VEH_SCO_CA1": "darkorchid",
    "VEH_SS_HILUS": "gainsboro",
    "VEH_SCO_HILUS": "gainsboro"
    
    
}




# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
#plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=1.0,
        label=category
    )

plt.title('UMAP Analysis by Group: CNEURO 1.0 SCO', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(UMAP_path + f"UMAP_{n_neighbors}_{min_dist}_Categories_CNEURO1_ESC .png", dpi=800)
plt.show()


######################################################################
######################################################################
#######################      VEH_SS_HILUS     ##############################
######################################################################
######################################################################


cluster_colors = {
    "VEH_SS_CA1": "gainsboro",
    "VEH_SCO_CA1": "gainsboro",
    "VEH_SS_HILUS": "limegreen",
    "VEH_SCO_HILUS": "gainsboro"
    
    
}


# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
#plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=1.0,
        label=category
    )

plt.title('UMAP Analysis by Group: VEH SCO', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(UMAP_path + f"UMAP_{n_neighbors}_{min_dist}_Categories_VEH_ESC.png", dpi=800)
plt.show()







######################################################################
######################################################################
#######################      VEH_SCO_HILUS     ##############################
######################################################################
######################################################################


cluster_colors = {
    "VEH_SS_CA1": "gainsboro",
    "VEH_SCO_CA1": "gainsboro",
    "VEH_SS_HILUS": "gainsboro",
    "VEH_SCO_HILUS": "crimson"
    
    
}


# Iterate over categories and plot points with respective colors
plt.figure(figsize=(12, 8))
#plt.style.use('dark_background')

for category, color in cluster_colors.items():
    category_data = data[data['categories'] == category]
    plt.scatter(
        embedding[category_data.index, 0],
        embedding[category_data.index, 1],
        color=color,
        alpha=1.0,
        label=category
    )

plt.title('UMAP Analysis by Group: CNEURO 0.1 SCO', fontsize=18)
plt.xlabel('UMAP 1', fontsize=14)
plt.ylabel('UMAP 2', fontsize=14)
plt.legend()
plt.grid(False)

plt.savefig(UMAP_path + f"UMAP_{n_neighbors}_{min_dist}_Categories_CNEURO-01_ESC.png", dpi=800)
plt.show()









######################################################################
######################################################################
######################################################################
######################################################################




######### JACCAR INDEX

jacard_plot =umap.plot.diagnostic(reducer, diagnostic_type='neighborhood')

# Save the plot using plt.savefig
plt.savefig(UMAP_path + f"Neighborhood_Jacard_{n_neighbors}_{min_dist}.png", dpi=800)

# Show the plot
plt.show()




######### Local Dimension

local_dims = umap.plot.diagnostic(reducer, diagnostic_type='local_dim')
# Save the plot using plt.savefig
plt.savefig(UMAP_path + f"Local_Dim_{n_neighbors}_{min_dist}.png", dpi=800)

# Show the plot
plt.show()




umap.plot.diagnostic(reducer, diagnostic_type='vq')
plt.savefig(UMAP_path + f"Vector_Quantization_{n_neighbors}_{min_dist}.png", dpi=800)

# Show the plot
plt.show()

######################################################################
######################################################################
######################################################################
######################################################################

######### CONNECTIVITY
"""
pip install numpy==1.23.4 



working fine with numpy==1.23.4

giving error with numpy==1.24.0

"""

connectivity_plot = umap.plot.connectivity(reducer, show_points=True, theme="inferno")
plt.savefig(UMAP_path + f"Connectivity_{n_neighbors}_{min_dist}_points.png", dpi=1000)
plt.show()


umap.plot.connectivity(reducer, edge_bundling='hammer', theme="inferno")
plt.savefig(UMAP_path + f"Connectivity_{n_neighbors}_{min_dist}_hammer.png", dpi=1000)
plt.show()




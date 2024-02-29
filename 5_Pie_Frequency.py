#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 21:18:36 2024

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
csv_path = set_path(o_path + "Merged_Data/")
Plot_path = set_path(o_path + "Plots/Descriptive/")



n_neighbors = 15
min_dist= 0.01
min_cluster_size= 20
min_samples=20

data = pd.read_csv(csv_path + f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv")

# Filter out rows where Cluster_Labels column has a value of -1
data = data[data['Cluster_Labels'] != -1]

# Convert Cluster_Labels column to string type
data['Cluster_Labels'] = data['Cluster_Labels'].astype(int)

######################################################################
######################################################################
######################################################################
######################################################################

###########################   PIE CHART   ############################


######################################################################
######################################################################
######################################################################
######################################################################




import matplotlib.pyplot as plt




cluster_colors = {
    0: "crimson",         
    1: "orangered",       
    4: "mediumturquoise", 
    5: "paleturquoise",   
    2: "gold",
    3: "limegreen",
    -1: "darkorchid"
}





# Group by 'categories' and 'Cluster', then count occurrences
grouped = data.groupby(['categories', 'Cluster_Labels']).size().reset_index(name='Count')

category_order = ["VEH_SS",  "VEH_SCO", "CNEURO-01_SCO", "CNEURO-10_SCO", "CNEURO-10_SS"]
category_labels = {"VEH_SS": "VEH SS", "CNEURO-10_SCO": "CNEURO 1.0 SCO", "VEH_SCO": "SCO", "CNEURO-01_SCO": "CNEURO 0.1 SCO", "CNEURO-10_SS": "CNEURO 1.0 SS"}

# Group the data further by 'categories'
grouped_by_category = grouped.groupby('categories')

# Create subplots for each category to represent pie charts
fig, axes = plt.subplots(1, len(grouped_by_category), figsize=(16, 8), facecolor='black')  # Set the background color

# Plot pie charts for each category in the specified order
for ax, category in zip(axes, category_order):
    if category in grouped_by_category.groups:
        category_data = grouped_by_category.get_group(category)
        category_data = category_data.set_index('Cluster_Labels')['Count']
        colors = [cluster_colors.get(cluster, 'grey') for cluster in category_data.index]

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return '{p:.2f}%'.format(p=pct)
            return my_autopct

        ax.set_xticklabels([category_labels.get(label, label) for label in category_order])
        ax.pie(category_data, labels=category_data.index, colors=colors, autopct=make_autopct(category_data.values), startangle=140)
        ax.set_title(f'{category_labels.get(category, category)}', fontweight='bold', color='white', fontsize=18)  # Use custom label for the title

        for text in ax.texts:
            text.set_fontweight('bold')
            text.set_color('white')  # Set text color
            text.set_fontsize(11) 

# Save and display the plot
plt.tight_layout()
plt.savefig(Plot_path + f"Pie_chart_Clusters_{n_neighbors}_{min_dist}.png", dpi=500, facecolor='black')  # Set the background color for the saved image
plt.show()







######################################################################
######################################################################
######################################################################
######################################################################

#######################   HISTOGRAM CLUSTERS   ########################


######################################################################
######################################################################
######################################################################
######################################################################



import seaborn as sns
import matplotlib.pyplot as plt


# Filter out rows with Cluster_Labels equal to -1
filtered_data = data[data['Cluster_Labels'] != -1]

# Set the style for the plots (optional)
sns.set(style="whitegrid")

# Create a count plot using seaborn with a custom color palette
plt.figure(figsize=(12, 8))  # Adjust the figure size if needed

# Set the background color to black
plt.style.use('dark_background')


# Specify the order of categories and corresponding labels
category_order = ["VEH_SS",  "VEH_SCO", "CNEURO-01_SCO", "CNEURO-10_SCO", "CNEURO-10_SS"]
category_labels = {"VEH_SS": "VEH SS", "CNEURO-10_SCO": "CNEURO 1.0 SCO", "VEH_SCO": "SCO", "CNEURO-01_SCO": "CNEURO 0.1 SCO", "CNEURO-10_SS": "CNEURO 1.0 SS"}
# Specify the order of clusters
cluster_order = [0,1,2,3,4,5]
# Use the custom palette and order in the countplot
ax = sns.countplot(x="categories", hue="Cluster_Labels", data=filtered_data, palette=cluster_colors, order=category_order)

# Customize x-axis labels
ax.set_xticklabels([category_labels.get(label, label) for label in category_order])

# Add title and labels
plt.title("Count of Cells for Each Cluster")
plt.xlabel("Categories")
plt.ylabel("Count of Cells")
plt.style.use('dark_background')

# Add legend
ax.legend(title='Clusters', loc='upper right', labels=[f'Cluster {i}' for i in cluster_order])

# Automatically adjust subplot parameters for better spacing
plt.tight_layout()

plt.savefig(Plot_path + f"Histogram_Clusters_{n_neighbors}_{min_dist}.png", dpi=800, bbox_inches="tight")

# Show the plot
plt.show()















import seaborn as sns
import matplotlib.pyplot as plt

# Filter out rows with Cluster_Labels equal to -1
filtered_data = data[data['Cluster_Labels'] != -1]

# Set the style for the plots (optional)
sns.set(style="whitegrid")

# Create a count plot using seaborn with a custom color palette
plt.figure(figsize=(12, 8))  # Adjust the figure size if needed

# Set the background color to black
plt.style.use('dark_background')

# Specify the order of clusters
cluster_order = [3,4,5,2,1,0]

# Use the custom palette and order in the countplot
ax = sns.countplot(x="categories", hue="Cluster_Labels", data=filtered_data, palette=cluster_colors, order=category_order, hue_order=cluster_order)

# Customize x-axis labels
ax.set_xticklabels([category_labels.get(label, label) for label in category_order])

# Add title and labels
plt.title("Count of Cells for Each Cluster")
plt.xlabel("Categories")
plt.ylabel("Count of Cells")
plt.style.use('dark_background')

# Add legend
ax.legend(title='Clusters', loc='upper right', labels=[f'Cluster {i}' for i in cluster_order])

plt.savefig(Plot_path + f"Histogram_Clusters_{n_neighbors}_{min_dist}_order.png", dpi=800, bbox_inches="tight")

# Show the plot
plt.show()


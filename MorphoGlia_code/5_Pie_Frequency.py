import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from morphoglia import set_path
import time

# Record start time
start_time = time.time()

# Load the data
i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "Merged_Data/")
Plot_path = set_path(o_path + "Plots/Descriptive/")

# Merge
n_neighbors = 10
min_dist = 0.1
min_cluster_size = 20
min_samples = 10

data = pd.read_csv(csv_path + f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv")

# Filter out rows where Cluster_Labels column has a value of -1
data = data[data['Clusters'] != -1]

# Convert Cluster_Labels column to string type
data['Clusters'] = data['Clusters'].astype(int)

# Define a mapping of old category names to new names
category_mapping = {
    "VEH_SS_CA1": "SS-CA1",
    "VEH_SCO_CA1": "SCOP-CA1",
    "VEH_SS_HILUS": "SS-Hilus",
    "VEH_SCO_HILUS": "SCOP-Hilus"
}

# Replace old category names with new names in the data
data['categories'] = data['categories'].replace(category_mapping)

# Define cluster colors
cluster_colors = {
    0: "darkorchid",
    1: "crimson",
    2: "gold",
    3: "limegreen",
    4: "mediumturquoise",
    -1: "white"
}

# Group by 'categories' and 'Cluster', then count occurrences
grouped = data.groupby(['categories', 'Clusters']).size().reset_index(name='Count')

category_order = ["SS-CA1", "SCOP-CA1", "SS-Hilus", "SCOP-Hilus"]
category_labels = {"SS-CA1": "SS-CA1", "SCOP-CA1": "SCOP-CA1", "SS-Hilus": "SS-Hilus", "SCOP-Hilus": "SCOP-Hilus"}

# Group the data further by 'categories'
grouped_by_category = grouped.groupby('categories')

# Create subplots for each category to represent pie charts
fig, axes = plt.subplots(1, len(grouped_by_category), figsize=(16, 8), facecolor='white')

# Plot pie charts for each category in the specified order
for ax, category in zip(axes, category_order):
    if category in grouped_by_category.groups:
        category_data = grouped_by_category.get_group(category)
        category_data = category_data.set_index('Clusters')['Count']
        colors = [cluster_colors.get(cluster, 'grey') for cluster in category_data.index]

        def make_autopct(values):
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return '{p:.2f}%'.format(p=pct)
            return my_autopct

        ax.set_xticklabels([category_labels.get(label, label) for label in category_order])
        ax.pie(category_data, labels=category_data.index, colors=colors, autopct=make_autopct(category_data.values), startangle=140)
        ax.set_title(f'{category_labels.get(category, category)}', fontweight='bold', color='black', fontsize=18)

        for text in ax.texts:
            text.set_fontweight('bold')
            text.set_color('white')
            text.set_fontsize(11)

# Save and display the plot
plt.tight_layout()
plt.savefig(Plot_path + f"Pie_chart_Clusters_{n_neighbors}_{min_dist}.png", dpi=500, facecolor='white')
plt.show()

# HISTOGRAM CLUSTERS

# Filter out rows with Cluster_Labels equal to -1
filtered_data = data[data['Clusters'] != -1]

# Set the style for the plots
sns.set(style="whitegrid")

# Create a count plot using seaborn with a custom color palette
plt.figure(figsize=(12, 8))

# Specify the order of clusters
cluster_order = [0, 1, 2, 3, 4]

# Use the custom palette and order in the countplot
ax = sns.countplot(x="categories", hue="Clusters", data=filtered_data, palette=cluster_colors, order=category_order)

# Customize x-axis labels
ax.set_xticklabels([category_labels.get(label, label) for label in category_order])

# Add title and labels
plt.title("Count of Cells for Each Cluster", fontweight='bold')
plt.xlabel("Categories", fontweight='bold')
plt.ylabel("Count of Cells", fontweight='bold')

# Add legend
ax.legend(title='Clusters', loc='upper right', labels=[f'Cluster {i}' for i in cluster_order])

# Automatically adjust subplot parameters for better spacing
plt.tight_layout()

plt.savefig(Plot_path + f"Histogram_Clusters_{n_neighbors}_{min_dist}.png", dpi=800, bbox_inches="tight")

# Show the plot
plt.show()






category_order = ["SCOP-Hilus", "SS-Hilus", "SCOP-CA1", "SS-CA1" ]

######################################################################
####################### Stacked Frequencies ##########################
######################################################################

# Create a stacked bar plot using seaborn with a custom color palette
plt.figure(figsize=(12, 8))

# Calculate the counts for each category and cluster
stacked_data = filtered_data.groupby(['categories', 'Clusters']).size().unstack(fill_value=0).reindex(category_order)

# Plot the stacked bar plot
stacked_data.plot(kind='barh', stacked=True, color=[cluster_colors.get(i, 'grey') for i in stacked_data.columns], ax=plt.gca())

# Customize y-axis labels
plt.gca().set_yticklabels([category_mapping.get(label, label) for label in category_order], rotation=0, fontweight='bold')

# Add title and labels
plt.title("Stacked Frequencies of Cells for Each Cluster", fontweight='bold')
plt.xlabel("Count of Cells", fontweight='bold')
plt.ylabel("Categories", fontweight='bold')

# Add legend
plt.legend(title='Clusters', loc='center left', bbox_to_anchor=(1, 0.5), labels=[f'Cluster {i}' for i in stacked_data.columns])

# Remove grid lines
plt.gca().grid(False)

# Customize x-axis tick labels to be bold
plt.gca().tick_params(axis='x', which='both', labelsize=12, labelcolor='black', width=2)
for label in plt.gca().get_xticklabels():
    label.set_fontweight('bold')

# Automatically adjust subplot parameters for better spacing
plt.tight_layout()

plt.savefig(Plot_path + f"Stacked_Histogram_Clusters_Frequencies_{n_neighbors}_{min_dist}.png", dpi=800, bbox_inches="tight")

# Show the plot
plt.show()

######################################################################
####################### Stacked Percentages ##########################
######################################################################

# Create a stacked bar plot using seaborn with a custom color palette
plt.figure(figsize=(12, 8))

# Calculate the percentages for each category and cluster
stacked_data_pct = stacked_data.div(stacked_data.sum(axis=1), axis=0)

# Plot the stacked bar plot with percentages
stacked_data_pct.plot(kind='barh', stacked=True, color=[cluster_colors.get(i, 'grey') for i in stacked_data_pct.columns], ax=plt.gca())

# Customize y-axis labels
plt.gca().set_yticklabels([category_mapping.get(label, label) for label in category_order], rotation=0, fontweight='bold')

# Add title and labels
plt.title("Stacked Percentages of Cells for Each Cluster", fontweight='bold')
plt.xlabel("Percentage of Cells", fontweight='bold')
plt.ylabel("Categories", fontweight='bold')

# Add legend
plt.legend(title='Clusters', loc='center left', bbox_to_anchor=(1, 0.5), labels=[f'Cluster {i}' for i in stacked_data_pct.columns])

# Remove grid lines
plt.gca().grid(False)

# Customize x-axis tick labels to be bold
plt.gca().tick_params(axis='x', which='both', labelsize=12, labelcolor='black', width=2)
for label in plt.gca().get_xticklabels():
    label.set_fontweight('bold')

# Automatically adjust subplot parameters for better spacing
plt.tight_layout()

plt.savefig(Plot_path + f"Stacked_Histogram_Clusters_Percentages_{n_neighbors}_{min_dist}.png", dpi=800, bbox_inches="tight")

# Show the plot
plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_min = elapsed_time / 60

print("")
print(f"Elapsed Time: {elapsed_time} seconds")
print(f"Elapsed Time: {elapsed_time_min} min")
print("")
print("________________________________________")
print("________________________________________")
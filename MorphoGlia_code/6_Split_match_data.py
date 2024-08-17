#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April  18 00:14:22 2024


@author: juanpablomayaarteaga
"""

import os
import pandas as pd



# Load the data
i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = os.path.join(i_path, "Output_images/")
csv_path = os.path.join(o_path, "Merged_Data")


#Merge
n_neighbors = 10
min_dist= 0.1
min_cluster_size= 20
min_samples=10

regions = ["HILUS", "CA1"]
subject = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
group = ["CNEURO-10", "VEH", "CNEURO-01"]
treatment = ["SCO", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

csv_file = f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"


data = os.path.join(csv_path, csv_file)

# Read the CSV file
df = pd.read_csv(data)



# Iterate over categories and create separate CSV files if the group exists
for region in regions:
    for s in subject:
        for g in group:
            for tr in treatment:
                for ti in tissue:
                    # Check if the combination exists in the DataFrame
                    combination_exists = not df[
                        (df['region'] == region) &
                        (df['subject'] == s) &
                        (df['group'] == g) &
                        (df['treatment'] == tr) &
                        (df['tissue'] == ti)
                    ].empty
                    
                    if combination_exists:
                        # Filter the dataframe based on the given criteria
                        filtered_data = df[
                            (df['region'] == region) &
                            (df['subject'] == s) &
                            (df['group'] == g) &
                            (df['treatment'] == tr) &
                            (df['tissue'] == ti)
                        ]
                        # Save the filtered data to a separate CSV file
                        output_file = f"{s}_{g}_{tr}_{ti}_{region}_Morphology_UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
                        output_path = os.path.join(csv_path, output_file)
                        filtered_data.to_csv(output_path, index=False)




print(output_path)

import os
import shutil

# Iterate over the CSV files and move each to its directory
for region in regions:
    for s in subject:
        for g in group:
            for tr in treatment:
                for ti in tissue:
                    csv_file_name = f"{s}_{g}_{tr}_{ti}_{region}_Morphology_UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
                    source_file = os.path.join(csv_path, csv_file_name)
                    target_directory = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_{region}_FOTO1_PROCESSED", "Data")
                    
                    if os.path.exists(source_file) and os.path.exists(target_directory):
                        shutil.copy2(source_file, target_directory)
                        os.remove(source_file)  # Remove the original file
    
                    else:
                        continue
                        print(f"File '{csv_file_name}' or directory '{target_directory}' does not exist.")


print(target_directory)
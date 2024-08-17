#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 18 00:14:22 2024 and Thu Jul 25 15:01:21 2024

@author: juanpablomayaarteaga
"""

import os
import pandas as pd
import shutil
import cv2
import numpy as np
import time
from morphoglia import set_path

# Load the data
i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = os.path.join(i_path, "Output_images/")
csv_path = os.path.join(o_path, "Merged_Data")
ID_path = set_path(os.path.join(o_path, "ID_clusters"))

n_neighbors = 10
min_dist = 0.1
min_cluster_size = 20
min_samples = 10

regions = ["HILUS", "CA1"]
subject = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
group = ["CNEURO-10", "VEH", "CNEURO-01"]
treatment = ["SCO", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

csv_file = f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
data = os.path.join(csv_path, csv_file)

# Read the CSV file
df = pd.read_csv(data)

# Start the timer
start_time = time.time()

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

# Move CSV files to their directories
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








regions = ["HILUS", "CA1"]
subject = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
group = ["CNEURO-10", "VEH", "CNEURO-01"]
treatment = ["SCO", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

df_color_position = pd.DataFrame(columns=["Cell", "Clusters", "x1", "y1", "x2", "y2"])

for region in regions:
    for s in subject:
        for g in group:
            for tr in treatment:
                for ti in tissue:
                    individual_img_path = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_{region}_FOTO1_PROCESSED/")
                    csv_path = os.path.join(individual_img_path, "Data/")
                    csv_file = f"{s}_{g}_{tr}_{ti}_{region}_Morphology_UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
                    original_img = f"{s}_{g}_{tr}_{ti}_{region}_FOTO1_PROCESSED.tif"

                    if os.path.isfile(os.path.join(csv_path, csv_file)):
                        print("Successfully loaded CSV file:", csv_file)
                        data = pd.read_csv(os.path.join(csv_path, csv_file))
                        
                        if os.path.exists(individual_img_path):

                            Cells_path = os.path.join(individual_img_path + "Cells/")
                            color_path = set_path(Cells_path + "Color_Cells/")
                            Cells_thresh_path = set_path(Cells_path + "Cells_thresh/")

                            if os.path.exists(Cells_thresh_path):
                                for _, row in data.iterrows():
                                    cell_number = row['Cell']
                                    cluster = row['Clusters']
                                    coordinates = eval(row['cell_positions'])
                                    x1, y1, x2, y2 = coordinates

                                    # Construct the image filename based on cell_number
                                    image_filename = f"{cell_number}.tif"

                                    if os.path.exists(Cells_thresh_path) and image_filename in os.listdir(Cells_thresh_path):
                                        input_image = os.path.join(Cells_thresh_path, image_filename)

                                        if os.path.isfile(input_image):
                                            # Read the grayscale image
                                            image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
                                            # Continue with image processing...

                                            # Apply thresholding
                                            threshold_value = 50  # Adjust this value as needed
                                            _, thresholded_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

                                            # Define cluster colors
                                            cluster_colors = {
                                                -1: (255, 255, 255),  # Purple
                                                0: (200, 0, 200),  # Purple
                                                1: (0, 0, 255),    # Red
                                                2: (0, 200, 200),  # Yellow
                                                3: (0, 255, 0),    # Green
                                                4: (220, 220, 0)   # Cyan
                                            }

                                            # Assign a color depending on its cluster number
                                            if cluster in cluster_colors:
                                                color = cluster_colors[cluster]
                                            else:
                                                color = (0, 0, 0)  # Default color if cluster is not found

                                            # Create a color image with an alpha channel
                                            colored_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

                                            # Apply the color to the thresholded regions using the mask
                                            colored_image[thresholded_mask == 255, :3] = color
                                            colored_image[thresholded_mask == 255, 3] = 255  # Set alpha to 255 for cell regions

                                            # Save the colored thresholded image with a unique filename
                                            output_image_path = os.path.join(color_path, f"{cell_number}.tif")
                                            cv2.imwrite(output_image_path, colored_image)

                                            # Add the cell information to the DataFrame
                                            df_color_position.loc[len(df_color_position)] = [f"{cell_number}.tif", cluster, x1, y1, x2, y2]
                    else:
                        continue

df_color_position = pd.DataFrame(columns=["Cell", "Clusters", "x1", "y1", "x2", "y2"])
for region in regions:
    for s in subject:
        for g in group:
            for tr in treatment:
                for ti in tissue:
                    # SAME NAME as images in PREPRO
                    original_img = f"{s}_{g}_{tr}_{ti}_{region}_FOTO1_PROCESSED.tif"
                    individual_img_path = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_{region}_FOTO1_PROCESSED/")
                    csv_path = os.path.join(individual_img_path, "Data/")
                    csv_file = f"{s}_{g}_{tr}_{ti}_{region}_Morphology_UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
                    
                    if os.path.isfile(os.path.join(i_path, original_img)):
                        original_image = cv2.imread(os.path.join(i_path, original_img))
                        print("Successfully loaded img:", original_image)
                        
                        if original_image is not None:

                            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2RGBA)
                            height, width, channels = original_image.shape
                            empty_image = np.zeros((height, width, 4), np.uint8)

                            if os.path.isfile(os.path.join(csv_path, csv_file)):
                                data = pd.read_csv(os.path.join(csv_path, csv_file))
                                
                                Cells_path = os.path.join(individual_img_path + "Cells/")
                                color_path = set_path(Cells_path + "Color_Cells/")

                                # Loop through the rows of the 'data' DataFrame
                                for _, row in data.iterrows():
                                    cell_number = row['Cell']
                                    cluster = row['Clusters']
                                    coordinates = eval(row['cell_positions'])
                                    x1, y1, x2, y2 = coordinates
                                    
                                    # Construct the image filename based on cell_number
                                    image_filename = f"{cell_number}.tif"
                                    
                                    # Check if the image file exists in the directory
                                    if image_filename in os.listdir(color_path):
                                        input_image = os.path.join(color_path, image_filename)
                                        
                                        # Check if the input image file exists
                                        if os.path.isfile(input_image):

                                            # Read the colored image with alpha channel
                                            colored_image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)

                                            # Calculate the dimensions of the region where the image will be pasted
                                            overlay_width = x2 - x1
                                            overlay_height = y2 - y1
                                            
                                            # Resize the colored image to fit the overlay region
                                            colored_image_resized = cv2.resize(colored_image, (overlay_width, overlay_height))

                                            # Overlay the resized colored image onto the empty image at the specified coordinates
                                            for c in range(0, 3):
                                                empty_image[y1:y2, x1:x2, c] = colored_image_resized[:, :, c] * (colored_image_resized[:, :, 3] / 255.0) + empty_image[y1:y2, x1:x2, c] * (1.0 - colored_image_resized[:, :, 3] / 255.0)
                                            empty_image[y1:y2, x1:x2, 3] = np.maximum(empty_image[y1:y2, x1:x2, 3], colored_image_resized[:, :, 3])
                                            
                                            # Draw a yellow rectangle around the pasted image
                                            cv2.rectangle(empty_image, (x1, y1), (x2, y2), (0, 255, 255, 255), 2)

                                            # Write the cell number at the top of the box in red
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            font_scale = 0.5
                                            font_color = (0, 0, 255, 255)
                                            font_thickness = 2
                                            text_size = cv2.getTextSize(str(cell_number), font, font_scale, font_thickness)[0]
                                            text_x = x1 + (overlay_width - text_size[0]) // 2
                                            text_y = y1 - 5
                                            cv2.putText(empty_image, str(cell_number), (text_x, text_y), font, font_scale, font_color, font_thickness)

                                # Save the resulting image
                                cluster_image_path = os.path.join(ID_path, f"{s}_{g}_{tr}_{ti}_{region}_Clusters.tif")
                                cv2.imwrite(cluster_image_path, empty_image)


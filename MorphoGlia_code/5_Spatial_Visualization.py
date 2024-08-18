#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:09:49 2024

@author: juanpablomaya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 18 00:14:22 2024
Updated by merging image processing and cluster analysis scripts

@author: juanpablomayaarteaga
"""

import os
import pandas as pd
import cv2
import numpy as np
import shutil
from morphoglia import set_path

# Paths
i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = os.path.join(i_path, "Output_images/")
csv_path = os.path.join(o_path, "Data/")
ID_path = set_path(o_path + f"ID_clusters/")

# Clustering Parameters
n_neighbors = 10
min_dist = 0.1
min_cluster_size = 20
min_samples = 10

regions = ["HILUS", "CA1"]
subject = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
group = ["CNEURO-10", "VEH", "CNEURO-01"]
treatment = ["SCO", "SS"]
tissue = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

# Load CSV Data
csv_file = f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
data_path = os.path.join(csv_path, csv_file)
df = pd.read_csv(data_path)

# Iterate through combinations to filter and save CSV files, then process images
for region in regions:
    for s in subject:
        for g in group:
            for tr in treatment:
                for ti in tissue:
                    # Check if the combination exists in the DataFrame
                    filtered_data = df[
                        (df['region'] == region) &
                        (df['subject'] == s) &
                        (df['group'] == g) &
                        (df['treatment'] == tr) &
                        (df['tissue'] == ti)
                    ]
                    
                    if not filtered_data.empty:
                        # Save the filtered data to a separate CSV file
                        output_file = f"{s}_{g}_{tr}_{ti}_{region}_Morphology_UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
                        output_path = os.path.join(csv_path, output_file)
                        filtered_data.to_csv(output_path, index=False)

                        # Move the CSV to the corresponding directory
                        target_directory = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_{region}_FOTO1_PROCESSED", "Data")
                        if os.path.exists(output_path) and os.path.exists(target_directory):
                            shutil.copy2(output_path, target_directory)
                            os.remove(output_path)
                        
                        # Start processing the corresponding images
                        individual_img_path = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_{region}_FOTO1_PROCESSED/")
                        Cells_path = set_path(os.path.join(individual_img_path, "Cells/"))
                        color_path = set_path(os.path.join(Cells_path, "Color_Cells/"))
                        Cells_thresh_path = set_path(os.path.join(Cells_path, "Cells_thresh/"))
                        original_img = os.path.join(i_path, f"{s}_{g}_{tr}_{ti}_{region}_FOTO1_PROCESSED.tif")

                        if os.path.isfile(original_img):
                            original_image = cv2.imread(original_img)
                            if original_image is not None:
                                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2RGBA)
                                height, width, channels = original_image.shape
                                empty_image = np.zeros((height, width, 4), np.uint8)

                                for _, row in filtered_data.iterrows():
                                    cell_number = row['Cell']
                                    cluster = row['Clusters']
                                    x1, y1, x2, y2 = eval(row['cell_positions'])

                                    # Construct the image filename based on cell_number
                                    image_filename = f"{cell_number}.tif"
                                    input_image_path = os.path.join(Cells_thresh_path, image_filename)

                                    if os.path.isfile(input_image_path):
                                        image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
                                        _, thresholded_mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

                                        # Define cluster colors
                                        cluster_colors = {
                                            -1: (255, 255, 255),  # White for noise
                                            0: (200, 0, 200),  # Purple
                                            1: (0, 0, 255),    # Red
                                            2: (0, 200, 200),  # Yellow
                                            3: (0, 255, 0),    # Green
                                            4: (220, 220, 0)   # Cyan
                                        }
                                        color = cluster_colors.get(cluster, (0, 0, 0))  # Default color if cluster not found

                                        # Create a color image with an alpha channel
                                        colored_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                                        colored_image[thresholded_mask == 255, :3] = color
                                        colored_image[thresholded_mask == 255, 3] = 255  # Set alpha to 255 for cell regions

                                        # Resize and overlay the colored image onto the original image
                                        overlay_width = x2 - x1
                                        overlay_height = y2 - y1
                                        resized_colored_image = cv2.resize(colored_image, (overlay_width, overlay_height))

                                        # Overlay the resized colored image onto the empty image
                                        for c in range(0, 3):
                                            empty_image[y1:y2, x1:x2, c] = resized_colored_image[:, :, c] * (resized_colored_image[:, :, 3] / 255.0) + empty_image[y1:y2, x1:x2, c] * (1.0 - resized_colored_image[:, :, 3] / 255.0)
                                        empty_image[y1:y2, x1:x2, 3] = np.maximum(empty_image[y1:y2, x1:x2, 3], resized_colored_image[:, :, 3])
                                        
                                        """
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
                                        """
                                            
                                            
                                # Save the resulting cluster visualization
                                cluster_image_path = os.path.join(ID_path, f"{s}_{g}_{tr}_{ti}_{region}_Clusters.tif")
                                cv2.imwrite(cluster_image_path, empty_image)
                                print(f"Cluster image saved at: {cluster_image_path}")

print("Processing completed.")

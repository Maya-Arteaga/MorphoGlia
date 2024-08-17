#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:11:01 2023

@author: juanpablomayaarteaga
"""


import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from morphoglia import set_path


# Load the data
i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = os.path.join(i_path, "Output_images/")
csv_path = os.path.join(o_path, "Merged_Data")

#Merge
n_neighbors = 10
min_dist= 0.1
min_cluster_size= 20
min_samples=10





ID_path= set_path(o_path+f"ID_clusters/")

#i_original_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/"



regions= ["HILUS", "CA1"]
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
        
            
                                        if image_filename in os.listdir(Cells_thresh_path):
                                            input_image = os.path.join(Cells_thresh_path, image_filename)
            
                                            if os.path.isfile(input_image):
                                                # Read the grayscale image
                                                image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
                                                # Continue with image processing...
            
                                                
            
                                            
                                                """
                                                bright_img = cv2.convertScaleAbs(image, alpha=3, beta=0)
                                                
                                                gaussian_array = filters.gaussian(bright_img, sigma=2.2)
                                                gaussian_image = (gaussian_array * 255).astype(np.uint8)
                                                gammaImg = gammaCorrection(gaussian_image, 1.25)
                                                """
                                                
                                                # Apply thresholding
                                                threshold_value = 50  # Adjust this value as needed
                                                _, thresholded_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
                                                
                                                """
                                                #NOISE
                                                cluster_colors = {
                                                 -1: (0, 0, 255), #White
                                                 #-1: (87, 139, 46),
                                                 0: (255, 255, 255),      #Purple
                                                 1: (255, 255, 255),      #Red
                                                 2: (255, 255, 255),    #Orange
                                                 3: (255, 255, 255),     #yellow
                                                 4: (255, 255, 255),    # green
                                                 5: (255, 255, 255),      # cyan
                                                 6: (255, 255, 255),   #palecyan
                                                 #: (255, 255, 255),

  
                                                 
                                              }
                                                
                                                
                                              
                                                """
                                                cluster_colors = {
                                                   #-1: (255, 255, 255), #White
                                                   #-1: (87, 139, 46),
                                                   0: (200, 0, 200),      #Purple
                                                   1: (0, 0, 255),      #Red
                                                   #2: (0, 100, 255),    #Orange
                                                   2: (0, 200, 200),     #yellow
                                                   3: (0, 255, 0),    # green
                                                   4: (220, 220, 0),      # cyan
                                                   #6: (245, 245, 190),   #palecyan
                                                   #: (255, 255, 255),
                                                   #6: (87, 139, 46),
    
                                                   
                                                }
                                                
    
                                                
                                                
                                            
                                                # Assign a color depending on its cluster number
                                                if cluster in cluster_colors:
                                                    color = cluster_colors[cluster]
                                                else:
                                                    color = (0, 0, 0)  # Default color if cluster is not found
                                    
                                                
                                                colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                                    
                                                # Apply the color to the thresholded regions using the mask
                                                colored_image[thresholded_mask == 255] = color
                                                
                                                # Convert BGR to BGRA and set the alpha channel for black background to 0
                                                rgba_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2BGRA)
                                                black_background = (colored_image[:, :, 0] == 0) & (colored_image[:, :, 1] == 0) & (colored_image[:, :, 2] == 0)
                                                rgba_image[:, :, 3][black_background] = 0
                                    
                                                # Save the colored thresholded image with a unique filename
                                                output_image_path = os.path.join(color_path, f"{cell_number}.tif")
                                                cv2.imwrite(output_image_path, rgba_image)
                                                
                                    
                                                # Add the cell information to the DataFrame
                                                df_color_position.loc[len(df_color_position)] = [f"{cell_number}.tif", cluster, x1, y1, x2, y2]
                                                #print("Successfully loaded stored:", output_image_path)
                    else:
                        continue
                                                




df_color_position = pd.DataFrame(columns=["Cell", "Clusters", "x1", "y1", "x2", "y2"])
for region in regions:
    for s in subject:
        for g in group:
            for tr in treatment:
                for ti in tissue:
                    #SAME NAME as images in PREPRO
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
                            empty_image = np.zeros((height, width, 3), np.uint8)
                            
                            #empty_image = np.zeros((height, width, 4), np.uint8)
                            #empty_image[:, :, 3] = 255
    
                            
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
                                    #print("Successfully loaded stored:", image_filename)
                                    
                                    
                                    # Check if the image file exists in the directory
                                    if image_filename in os.listdir(color_path):
                                        input_image = os.path.join(color_path, image_filename)
                                        
                                        # Check if the input image file exists
                                        if os.path.isfile(input_image):
    
                                            # Read the colored image
                                            colored_image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
                                            colored_image = colored_image[:, :, :3]
                                            #rgba_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2BGRA)
                                            #black_background = (colored_image[:, :, 0] == 0) & (colored_image[:, :, 1] == 0) & (colored_image[:, :, 2] == 0)
                                            #rgba_image[:, :, 3][black_background] = 0
                                            #plt.imshow(rgba_image)
                                            
                                            # Calculate the dimensions of the region where the image will be pasted
                                            overlay_width = x2 - x1
                                            overlay_height = y2 - y1
                                            
                                            # Resize the colored image to fit the overlay region
                                            colored_image_resized = cv2.resize(colored_image, (overlay_width, overlay_height))
                                            
                                            # Overlay the resized colored image onto the empty image at the specified coordinates
                                            #empty_image[y1:y2, x1:x2] = colored_image_resized
                                            
                                            # Ensure the colored image is resized to fit the area you want to overlay
                                            colored_image_resized = cv2.resize(colored_image_resized, (x2 - x1, y2 - y1))
                                            
                                            # Overlay the images using alpha blending
                                            alpha = 0.3  # Adjust the alpha value as needed for transparency
                                            
                                            # Create a region of interest (ROI) on the empty image for overlay
                                            roi = empty_image[y1:y1 + colored_image_resized.shape[0], x1:x1 + colored_image_resized.shape[1]]
                                            
                                            # Perform the overlay using alpha blending
                                            cv2.addWeighted(roi, alpha, colored_image_resized, 1 - alpha, 0, roi)
                                            
                                            
                                            
                                            # Draw a yellow rectangle around the pasted image
                                            cv2.rectangle(empty_image, (x1, y1), (x2, y2), (0, 255, 255, 0), 2)
                                            
                                            # Write the cell number at the top of the box in red
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            font_scale = 0.5
                                            font_color = (0, 0, 255)
                                            font_thickness = 2
                                            text_size = cv2.getTextSize(str(cell_number), font, font_scale, font_thickness)[0]
                                            text_x = x1 + (overlay_width - text_size[0]) // 2
                                            text_y = y1 - 5
                                            cv2.putText(empty_image, str(cell_number), (text_x, text_y), font, font_scale, font_color, font_thickness)
                                
                                # Save the resulting image
                                cluster_image_path = os.path.join(ID_path, f"{s}_{g}_{tr}_{ti}_{region}_Clusters.tif")
                                cv2.imwrite(cluster_image_path, empty_image)
    
                                        
        
        
        
    
    
    
                                

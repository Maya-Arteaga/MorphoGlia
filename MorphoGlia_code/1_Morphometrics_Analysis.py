#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File nomenclature:

R3_CNEURO1_ESC_T7_CA1_FOTO1_PROCESSED.tif    
R3_VEH_SS_T3_CA1_FOTO1_PROCESSED.tif         
R4_VEH_ESC_T10_CA1_FOTO1_PROCESSED.tif       
R7_CNEURO-01_ESC_T8_CA1_FOTO1_PROCESSED.tif

To rename the nomenclature it was used zsh:
Changing ESC to SCO
    
autoload -U zmv
zmv '(*_ESC_*)' '${1//_ESC_/_SCO_}'


changin CNEURO1 to CNEURO-10
autoload -U zmv
zmv '(*_ESC_*)' '${1//_ESC_/_SCO_}' && zmv '(*_CNEURO1_*)' '${1//_CNEURO1_/_CNEURO-10_}'


or doing both at the same time

autoload -U zmv
zmv '(*_ESC_*)' '${1//_ESC_/_SCO_}' && zmv '(*_CNEURO1_*)' '${1//_CNEURO1_/_CNEURO-10_}'


"""



import os
import cv2
import pandas as pd
import numpy as np
import morphoglia as mg
import tifffile as tiff
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import time
from skimage import io

# Record start time
start_time = time.time()



#Set and create paths 
i_path="/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path= mg.set_path(i_path+"Output_images/")
ID_path= mg.set_path(o_path+"ID/")
merge_csv_path = mg.set_path(o_path + "/Data/")
Plot_path = mg.set_path(o_path + "Plots/")


#From the preprocessed images, extract the nomenclature 
regions= ["HILUS", "CA1"]
subjects = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
groups = ["CNEURO-10", "VEH", "CNEURO-01"]
treatments = ["SCO", "SS"]
tissues = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

           
#For loop to process all the TIF files in the 
for region in regions:
    for subject in subjects:
        for group in groups:
            for treatment in treatments:
                for tissue in tissues:
                     
                     # READING THE INDIVIDUAL TIF FILES OF THE DIRECTORY
                     
                     img = f"{subject}_{group}_{treatment}_{tissue}_{region}_FOTO1_PROCESSED"
                     tif= ".tif"
    
                    
                     image_path = i_path + img + tif
                    
                    
                     if os.path.isfile(image_path):
                         
                        print("Reading image:", image_path)
                        # SETTING NEW PATHS (THEREFORE DIRECTORIES) TO STORE EACH ANALYSIS 
                        # AND SUPERVISED THAT ALL THE METRICS ARE WELL PROCESSED
                        
                        print ("...")
                        print ("...")
                        print("Setting paths...")
                        print ("...")
                        print ("...")
                        
                        #Setting directories for each image. This let us to handle each image and supervised each cell substracted from each image:
                        
                        individual_img_path= mg.set_path(o_path+f"{subject}_{group}_{treatment}_{tissue}_{region}_FOTO1_PROCESSED/")
                        
                        Cells_path= mg.set_path(individual_img_path+"Cells/")
                        
                        Cells_thresh_path= mg.set_path(Cells_path+"Cells_thresh/")
                        
                        Soma_path= mg.set_path(Cells_path+"Soma/")
                        
                        Skeleton_path= mg.set_path(Cells_path+"Skeleton/")
                        
                        Branches_path=mg.set_path(Cells_path+"Branches/")
                        
                        Skeleton2_path= mg.set_path(Cells_path+"Skeleton2/")
    
                        Branches2_path= mg.set_path(Cells_path+"Branches2/")
                        
                        Skeleton_Soma_path= mg.set_path(Cells_path+"Skeleton_Soma/")
                        
                        Soma_centroid_path= mg.set_path(Cells_path+"Soma_Centroids/")
                        
                        Convex_Hull_path= mg.set_path(Cells_path+"Convex_Hull/")
                        
                        Convex_Hull_Centroid_path= mg.set_path(Cells_path+"Convex_Hull_Centroid/")
                        
                        Branches3_path= mg.set_path(Cells_path+"Branches3/")
                        
                        Cell_centroid_path= mg.set_path(Cells_path+"Cell_centroid/")
                        
                        Sholl_path= mg.set_path(Cells_path+"Sholl/")
                        
                        csv_path= mg.set_path(individual_img_path+"Data/")
                        
                        
                        #Reading the images as gray
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        
                        #Invert binary: in the preproccesed images the foreground objects are in black and the background in white
                        #so here we are inverting the foreground to white and background to black
                        image = cv2.bitwise_not(image)
                        
                        
                        #THE IMAGES ARE ALREADY PREPROCESSED AND THERSHOLDED
                        thresh = image
    
                        # LABEL OBJECTS
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
                        
                        # SET PARAMETERS TO EXCLUDE NOISE 
                        min_area = 1000
                        max_area = 24000000
                        #max_area=6000
                        num_cells_filtered = 0
                       
                  
                        #### 
                        print ("...")
                        print ("...")
                        print("Identifying individual cells ORIGINAL...")
                        print ("...")
                        print ("...")
                        
                        
                        # Loop to get individual cells from threshold image
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert image to BGR color format
                        #plt.imshow(image)
                
                        individual_cell=0
                        
                        # Initialize the list to store cell positions
                        cell_positions = []
                        cell_num = []
                        
                        
                        for i in range(1, num_labels):
                            area = stats[i, cv2.CC_STAT_AREA]
                            if area < min_area or area > max_area:
                                labels[labels == i] = 0
                           
                            else:                
                                bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
                                bounding_box_y = stats[i, cv2.CC_STAT_TOP]
                                bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
                                bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]
                                cell_positions.append((bounding_box_x, bounding_box_y, bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height))
                                
                                #Individual cells
                                object_img = image[bounding_box_y:bounding_box_y + bounding_box_height, bounding_box_x:bounding_box_x + bounding_box_width]
                                individual_cell += 1
                                cell_num.append(individual_cell)
                                
                                #Save individual cells   
                                output_filename = f"{img}_cell_{individual_cell}.tif"
                                output_path = os.path.join(Cells_thresh_path, output_filename)
                                tiff.imwrite(output_path, object_img)
                                
                                
                        
                        
                        # Save the threshold image with red rectangles around objects
                        #save_tif(images, name="_rectangles.tif", path=Label_path, variable=image)
                        print ("...")
                        print ("...")
                        print("The cells have been sucessfully identified and subtracted from the image")
                        print ("...")
                        print ("...")
    
                        # Till here, the paths should be created and the directory "Cell_thresh", 
                        # which is whithin the directory Output_images/format_name_of_your_images/Cells/Cells_thresh
                        # should contain all the substracted cell from the images. 
                        # The other directories should be empty
    
    ###############################################################################
    ####################################################################################
    ###############################################################################################
    ####################################################################################################            
    ##########################################################################################
    ###############################################################################      
        
    
        
                # IDENTIFYING THE CELLS IN THE THERSHOLDED IMAGE
                # CREATING AN ID TO EACH CELL
                       
                        color_image = image
                        labeled_cell2 = 0
                        # Iterate over each labeled object
                        for i in range(1, num_labels):
                            area = stats[i, cv2.CC_STAT_AREA]
                            if area < min_area or area > max_area:
                                labels[labels == i] = 0
                            else:
                                bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
                                bounding_box_y = stats[i, cv2.CC_STAT_TOP]
                                bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
                                bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]
                                # Draw a rectangle around the object in the original image
                                cv2.rectangle(color_image, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 255, 0), 2)
                                labeled_cell2 += 1
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                bottom_left = (bounding_box_x, bounding_box_y + bounding_box_height + 20)
                                font_scale = 0.5
                                color = (255, 0, 0)
                                thickness = 2
                                cv2.putText(color_image, str(labeled_cell2), bottom_left, font, font_scale, color, thickness)
                        
                        # Save the original image with red rectangles around objects
                        mg.save_tif(output_filename, name=".tif", path=ID_path, variable=color_image)
                
                
                        print ("...")
                        print ("...")
                        print("Reference image ready!")
                        print ("...")
                        print ("...")
                        
                        
                        # CAPTURING THE POSITION OF THE CELLS IN ITS ORIGINAL IMAGE
                        mg.name_to_number(Cells_thresh_path)
                        df_positions = pd.DataFrame({'Cell': cell_num, 'cell_positions': cell_positions})
                        
                        #CREATING A DF WITH THE CELL POSITIONS
                        # Sort the DataFrame by the "Cell" column as numeric values
                        df_positions = df_positions.sort_values(by='Cell', key=lambda x: x.astype(int))
                        df_positions.to_csv(csv_path + "Cell_Positions.csv", index=False)
    
                       
                        #Till here, the directory "ID"
                        #which is in Output_images/ID
                        #must contain the same images as you are processing with the suffix of the number of cels identified in each image
                        #with the cells identified with a boundary rectangle
                        #This way, you can inspect which cells are identified an its identifier number 
                        
                       
    
                    
                        
    
    
    
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    ##############################################  REMOVE OBJECTS ##################################################
    
    
    
    
                        ### SUBSTACTING OBJECTS NOT CORRESPONDING TO THE CELLS 
                        # (BRANCESS FROM OTHER CELLS THAT WERE CLOSE TO THE CELL AND, THEREFORE
                        # CAPTURED BY THE BOX DELIMITATING THE CELL)
                        
                        print ("...")
                        print ("...")
                        print("Substracting OBJECTS NOT CORRESPONDING TO THE CELLS...")
                        print ("...")
                        print ("...")
                        
                        for images in os.listdir(Cells_thresh_path):
                            if images.endswith(".tif"):
                                
                                input_file = os.path.join(Cells_thresh_path, images)
                                image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    
                                # Find connected components and label them
                                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
                                # Find the label with the largest area (THE CELL)
                                largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Exclude background label (0)
                                
                                # Create a mask to keep only the largest object
                                mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                                # Apply the mask to the original image to keep only the largest object
                                result = cv2.bitwise_and(image, image, mask=mask)
                                mg.save_tif(images, name=".tif", path=Cells_thresh_path, variable=result)
                            
                            
                            
                            
                            
                            else:
                                continue
                                
                                
    
    
                        
                        print ("...")
                        print ("...")
                        print("NOISE REMOVED!")
                        print ("...")
                        print ("...")
                        
                        
                        
                        #This chunck of code removes objects that do not belong to the cells
                        #Note that if there are some branches that are not connected to the cells, this are going
                        #to be taken as external objects and are going to be removed. Be careful.
    
    
    
    
    ##############################################  REMOVE OBJECTS ##################################################
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    
    
    
    
    
    
    
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    #########################################  CELL FEATURES ANALYSIS #############################################
    
    
                        # FIRST ANALYSIS: FEATURES OF THE CELL
                        # DF OF THRESHOLDED CELL FEATURES: AREA, PERIMETER, CIRCULARITY, ...
                                     
                        print ("...")
                        print ("...")
                        print("Substracting CELL FEATURES...")
                        print ("...")
                        print ("...")
                        
                        # Initialize a list to collect data for all cells
                        data_list = []
                        
                        # Cell Area, Perimeter, and Circularity
                        for cell in os.listdir(Cells_thresh_path):
                            if cell.endswith(".tif"):
                                thresh_cell_path = os.path.join(Cells_thresh_path, cell)
                                
                                # Extract features for the current cell
                                Cell_area, Cell_perimeter, Cell_circularity, Cell_compactness, Cell_orientation, Cell_feret_diameter, Cell_eccentricity, Cell_aspect_ratio = mg.cell_analysis(thresh_cell_path)
                                
                                # Append the results to the list
                                data_list.append({
                                    "Cell": cell, 
                                    "Cell_area": Cell_area, 
                                    "Cell_perimeter": Cell_perimeter, 
                                    "Cell_circularity": Cell_circularity, 
                                    "Cell_compactness": Cell_compactness, 
                                    "Cell_orientation": Cell_orientation, 
                                    "Cell_feret_diameter": Cell_feret_diameter, 
                                    "Cell_eccentricity": Cell_eccentricity, 
                                    "Cell_aspect_ratio": Cell_aspect_ratio
                                })
                        
                        # Convert the list of dictionaries to a DataFrame
                        df_cell = pd.DataFrame(data_list)
                        
                        # Extract the numeric part from the "Cell" column and convert it to integers
                        df_cell['Cell'] = df_cell['Cell'].str.extract(r'(\d+)').astype(int)
                        
                        # Sort the DataFrame by the "Cell" column in ascending order
                        df_cell = df_cell.sort_values(by='Cell')
                        
                        # Save the DataFrame to a CSV file
                        df_cell.to_csv(csv_path + "Cell_features.csv", index=False)
                        
                        print ("...")
                        print ("...")
                        print("CELL FEATURES substracted!")
                        print ("...")
                        print ("...")

    
                        #Till here, you should have 2 csv files within the directory "DATA"
                        # Output_images/format_name_of_your_images/Data/
                        # thesde files are "Cell_features.csv" and "Cell_Positions.csv"
                        # Check that the values are not repeated, to be sure that all the cells are being processed correctly
                        
                        # Till here, no new images have been genereted for the directories in Output_images/format_name_of_your_images/Cells/
                        
                        
    #########################################  CELL FEATURES ANALYSIS #############################################
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    
    
    
    
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    #########################################  CELL SOMA ANALYSIS #################################################
    
                        ### SECOND ANALYSIS: SOMA OF THE CELL
                        # SUBSTRACTING THE SOMA OF THE CELL: CALCULATING ITS CIRCULARITY, AREA, PERIMETER, ...
                        # Here we are going to generete the images for the soma directorie in Output_images/format_name_of_your_images/Cells/Soma
             
                
                        # FIRST, THE SOMA IS SUBSTRACTED FROM THE THERSHOLDED CELL
                        # SECOND, THE FEATURES OF THE SOMA AREA ANLYZED
               
                        print ("...")
                        print ("...")
                        print("Substracting soma of the cells...")
                        print ("...")
                        print ("...")
                        
                        for images in os.listdir(Cells_thresh_path):
                            if images.endswith(".tif"):
                                input_file = os.path.join(Cells_thresh_path, images)
                                image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
                                
                                
                                # Opening preprocessing to dealing with the branches
                                kernel = np.ones((3,3),np.uint8)
                                
                                #REMOVING BRANCHES TO JUST ANALYSE THE SOMA: ERODE AND LARGEST OBJECT
                                eroded = cv2.erode(image, kernel, iterations=4)
                                # Find connected components and label them
                                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
                                # Find the label with the largest area
                                largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1  # Exclude background label (0)
                                # Create a mask to keep only the largest object
                                mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                                # Apply the mask to the original image to keep only the largest object
                                result = cv2.bitwise_and(eroded, eroded, mask=mask)
                                
                                #RESTORING THE ORIGINAL SIZE THAT WAS ERODE
                                dilated = cv2.dilate(result, kernel, iterations=4)
                                
                                # Get image dimensions
                                height, width = dilated.shape[:2]
                                
                                # Define the size of the frame
                                frame_thickness = 10
                                
                                # Create a mask to identify the frame area
                                mask = np.ones(dilated.shape[:2], dtype="uint8") * 255  # Initialize a white mask
                                
                                # Set the top row, bottom row, and both sides of the frame to black in the mask
                                mask[:frame_thickness, :] = 0  # First row at the top
                                mask[height - frame_thickness:, :] = 0  # Bottom row at the bottom
                                mask[:, :frame_thickness] = 0  # First column at the left
                                mask[:, width - frame_thickness:] = 0  # Last column at the right
                                
                                # Apply the mask to the image
                                result = cv2.bitwise_and(dilated, dilated, mask=mask)
                                
                                # Save the image as a tiff file
                                mg.save_tif(images, name=".tif", path=Soma_path, variable=result)
                            
                            else:
                                continue
                                
                                
                        mg.name_to_number(Soma_path)
    
                        
                        print ("...")
                        print ("...")
                        print("Soma substracted!")
                        print ("...")
                        print ("...")
                        
                        
                        #Til here, the directory "Soma" must contain the images of the soma of the cells
                        
    
    ################################################# SOMA ANALYSIS #################################################
    
    
    
                        print ("...")
                        print ("...")
                        print("ANALYZING Soma features...")
                        print ("...")
                        print ("...")
                        
                        # Initialize a list to collect data for all soma features
                        data_list = []
                        
                        # Cell Area, Perimeter, and Circularity
                        for soma in os.listdir(Soma_path):
                            if soma.endswith(".tif"):
                                cell_soma_path = os.path.join(Soma_path, soma)
                                
                                # Analyze soma features
                                Soma_area, Soma_perimeter, Soma_circularity, Soma_compactness, Soma_orientation, Soma_feret_diameter, Soma_eccentricity, Soma_aspect_ratio = mg.soma_analysis(cell_soma_path)
                                
                                # Append the results to the list
                                data_list.append({
                                    "Cell": soma,
                                    "Soma_area": Soma_area,
                                    "Soma_perimeter": Soma_perimeter,
                                    "Soma_circularity": Soma_circularity,
                                    "Soma_compactness": Soma_compactness,
                                    "Soma_orientation": Soma_orientation,
                                    "Soma_feret_diameter": Soma_feret_diameter,
                                    "Soma_eccentricity": Soma_eccentricity,
                                    "Soma_aspect_ratio": Soma_aspect_ratio
                                })
                        
                        # Convert the list of dictionaries to a DataFrame
                        df_area = pd.DataFrame(data_list)
                        
                        # Extract the numeric part from the "Cell" column and convert it to integers
                        df_area['Cell'] = df_area['Cell'].str.extract(r'(\d+)').astype(int)
                        
                        # Sort the DataFrame by the "Cell" column in ascending order
                        df_area = df_area.sort_values(by='Cell')
                        
                        # Save the DataFrame to a CSV file
                        df_area.to_csv(csv_path + "Soma_features.csv", index=False)
                        
                        print ("...")
                        print ("...")
                        print("Soma features ANALYZED!")
                        print ("...")
                        print ("...")

    
    
    
    
                        #Till here, you should have 3 csv files within the directory "DATA"
                        # Output_images/format_name_of_your_images/Data/
                        # thesde files are "Cell_features.csv" and "Cell_Positions.csv" and "Soma_features.csv"
                        # Check that the values are not repeated, to be sure that all the cells are being processed correctly
                        
                        # Till here, "CEll_Thresh" and "Soma" directories in Output_images/format_name_of_your_images/Cells/ contain images
                        
    
    
    #########################################  CELL SOMA ANALYSIS #################################################
    ###############################################################################################################
    ###############################################################################################################
    
    
    
    
    
    
    ###############################################################################################################
    ###############################################################################################################
    ################################################# ANALSIS SKELETONIZE #########################################
    
    
    
    # ANALISIS TRADICIONAL DE ZHAN DE SKELETONIZE QUE SE USA EN IMAGEJ PARA MORFOLOGIA CON MEJORAS
                        
                        ### THIRD ANALYSIS: BRANCHES OF THE CELL
                        # SUBSTRACTING THE BRANCHES OF THE CELL: CALCULATING ITS END POINTS, JUNCTIONS POINTS, ...
               
                  
                        # FIRST, THE CELL IS THINNING FROM THE THERSHOLDED CELL
                        # TO OBTAIN THE SKELETON OF THE CELL
                        # SECOND, THE FEATURES OF THE SKELETON AREA ANLYZED
                        # THE REGION CORRESPONDING TO THE SOMA IS SUBSTRACTED
                        # TO ENSURE THE ONLY ANALYSIS OF THE BRANCHES AND TO OBTAIN 
                        # THE CELL PROCESS (HERE DENOMINATED "INITIAL POINTS")
                    
    
        
                        print ("...")
                        print ("...")
                        print("Substracting Cell Ramifications features...")
                        print ("...")
                        print ("...")
                        
                        
    
                        
                        
                        #Skeletonize and detect branches
                        
                        for cell in os.listdir(Cells_thresh_path):
                            if cell.endswith(".tif"):
                                input_cell = os.path.join(Cells_thresh_path, cell)
                                cell_img = cv2.imread(input_cell, cv2.IMREAD_GRAYSCALE)
                                scale = cell_img /255
                                
                                #Skeletonize
                                skeleton = skeletonize(scale)
                                clean_skeleton= mg.erase(skeleton, 40)
                                mg.save_tif(cell, name=".tif", path=Skeleton_path, variable=clean_skeleton)
                                
                                #Detecteding Branches features
                                M, colored_image= mg.detect_and_color(clean_skeleton)
                                mg.save_tif(cell, name=".tif", path=Branches_path, variable=colored_image)
                         
                            
                        mg.name_to_number(Branches_path)    
                        
    
    
    
    ################################################# SKELETON WITHOUT SOMA ##########################################
    
    
    
                        print ("...")
                        print ("...")
                        print("Getting the Skeleton whitout Soma")
                        print ("...")
                        print ("...")
                        
                        # GEETTING THE SKELETON WITHOUT THE SOMA
    
                        for image in os.listdir(Skeleton_path):
                            if image.endswith(".tif"):
                                input_skeleton = os.path.join(Skeleton_path, image)
                                input_soma = os.path.join(Soma_path, image)
                        
                                skeleton_img = cv2.imread(input_skeleton, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale
                                soma_img = cv2.imread(input_soma, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale
                        
                                # Check if the images have the same dimensions
                                if skeleton_img.shape == soma_img.shape:
                                    subtracted_image = cv2.subtract(skeleton_img, soma_img)
                                    mg.save_tif(image, ".tif", Skeleton2_path, subtracted_image)
                        
                                    # Detecting branches features
                                    M, colored_image = mg.detect_and_color(subtracted_image)
                                    
                                    #Saving images in BRanches2
                                    #these are the colored skeleton without the soma 
                                    mg.save_tif(image, name=".tif", path=Branches2_path, variable=colored_image)
                        
                                    added_image = cv2.add(skeleton_img, soma_img)
                                    #Saving images of the skeleton with the soma as foreground objects (white or binary)
                                    mg.save_tif(image, ".tif", Skeleton_Soma_path, added_image)
                        
                         
                        
                        
                        print ("...")
                        print ("...")
                        print("SKELETON whitout Soma substracted!")
                        print ("...")
                        print ("...")
    
    
    
    ################################################# ESQUELETO SIN SOMA ##########################################                  
    
    
    
    
    ##############################################  INITIAL POINTS IMAGES ##################################
                     
                        
                        print ("...")
                        print ("...")
                        print("Generating INITIAL POINTS and Skeleton analysis...")
                        print ("...")
                        print ("...")
                        
                        #GENERETTING IMAGES FOR BRANCHES3 DIRECTORY. THESE ARE THE IMAGES THAT CONTAINS THE INITIAL POINTS
                        
                        for image in os.listdir(Skeleton_Soma_path):
                            if image.endswith(".tif"):
                                input_skeleton = os.path.join(Skeleton_Soma_path, image)
                                input_soma = os.path.join(Soma_path, image)
                        
                                skeleton_img = cv2.imread(input_skeleton, cv2.IMREAD_GRAYSCALE)
                                soma_img = cv2.imread(input_soma, cv2.IMREAD_GRAYSCALE)
                        
                                # Check if the images have the same dimensions
                                if skeleton_img.shape == soma_img.shape:
                                    subtracted_image = cv2.subtract(skeleton_img, soma_img)
                                    #save_tif(image, "_branch_.tif", Branches3_path, subtracted_image)
                        
                                    #Detecting branches features
                                    M, colored_image3 = mg.detect_features(subtracted_image, soma_img)
                                    mg.save_tif(image, name=".tif", path=Branches3_path, variable=colored_image3)
    
    
    ##############################################  INITIAL POINTS IMAGES ##################################
       
    
    
    ############################################## SKELETON WITHOUT SOMA ANALYSIS ##################################                         
                        
                        
                        df_skeleton = pd.DataFrame(columns=["Cell", "End_Points", "Junctions", "Branches", "Initial_Points", "Total_Branches_Length"])  # Initialize DataFrame with column names
                        
                        
                        
                        # ANALYZING BRANCHES FEATURES
     
                        print ("...")
                        print ("...")
                        print("Analyzing the SKELETON features...")
                        print ("...")
                        print ("...")
                        
                        
                        for skeleton in os.listdir(Branches3_path):
                            if skeleton.endswith(".tif"): 
                                skeleton_image_path = os.path.join(Branches3_path, skeleton)
                                
                                skeleton_img = io.imread(skeleton_image_path)
                                #plt.imshow(cell_img)
                                        
                                End_points = skeleton_img[:, :, 0] == 255
                                num_end_points = mg.count(End_points)
                                #plt.imshow(End_points)
                                        
                                Junction_points = skeleton_img[:, :, 1] == 255
                                num_junction_points = mg.count(Junction_points)
                                #plt.imshow(Junction_points)
                                
                                Length = skeleton_img[:, :, 2]
                                branches_length = mg.count(Length) + num_end_points
                                
                                        
                                Branches = skeleton_img[:, :, 2]
                                num_branches = mg.count_branches(Branches)
                                #plt.imshow(Branches)
                                
                                Initial_points = skeleton_img[:,:,1] == 200
                                # Count the initial points
                                num_initial_points = mg.count(Initial_points)
                                #plt.imshow(Initial_points)
                                       
                                
                                # Append the results to the DataFrame
                                df_skeleton.loc[len(df_skeleton)] = {
                                    "Cell": skeleton, 
                                    "End_Points": num_end_points, 
                                    "Junctions": num_junction_points, 
                                    "Branches": num_branches, 
                                    "Initial_Points": num_initial_points, 
                                    "Total_Branches_Length": branches_length
                                }
    
    
                        
                        # Extract numerical part from the "Cell" column
                        df_skeleton['Cell'] = df_skeleton['Cell'].str.extract(r'(\d+)').astype(int)    
                        
                        # Sort the DataFrame by the "Cell" column in ascending order
                        df_skeleton = df_skeleton.sort_values(by='Cell')
                        #df_area['Cell'] = df_area['Cell'].astype(str)
                        
                        # Calculate 'ratio_branches'
                        df_skeleton['ratio_branches'] = df_skeleton['End_Points'] / df_skeleton['Initial_Points']
                        column_name = 'ratio_branches'
                        
                        # Save the DataFrame to a CSV file
                        df_skeleton.to_csv(csv_path + "Skeleton_features.csv", index=False)
    
    
    
                        print ("...")
                        print ("...")
                        print("Skeleton features detected")
                        print ("...")
                        print ("...")
    
    
                        
    
    
    ############################################## SKELETON WITHOUT SOMA ANALYSIS ##################################
    
    
    
    
    ################################################# SKELETON ANALYSIS #########################################
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    ################################################# SHOLL ANALYSIS ###################################################
    
    
                        
                        print ("...")
                        print ("...")
                        print("Getting Cell centroids...")
                        print ("...")
                        print ("...")
                        
                        
                        centroids_soma = []
                        #df_sholl= pd.DataFrame(columns=["Cell", "centroids_soma", "sholl_max_dist", "sholl_circles", "sholl_crossing_points"])
                        for images in os.listdir(Soma_path):
                            if images.endswith(".tif"):
                                input_file = os.path.join(Soma_path, images)
                                img = io.imread(input_file)
                                
                                # Find contours in the binary image
                                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if len(contours) > 0:
                                    # Take the first (largest) contour
                                    largest_contour = max(contours, key=cv2.contourArea)
                                
                                    # Calculate the moments of the contour
                                    M = cv2.moments(largest_contour)
                                
                                    if M['m00'] != 0:
                                        # Calculate the centroid coordinates
                                        centroid_x = int(M['m10'] / M['m00'])
                                        centroid_y = int(M['m01'] / M['m00'])
                                        centroid = (centroid_x, centroid_y)
                                        centroids_soma.append((centroid_x, centroid_y))
                                        
                                
                                        print(f"Centroid coordinates: {centroid_x}, {centroid_y}")
                                    else:
                                        print("Object has no area (m00=0)")
                                else:
                                    print("No contours found in the image")
                                    
                                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                                
                                # Draw the centroid on the image for visualization
                                centroid_img= cv2.circle(img_color, centroid, 1, (255, 0, 0), -1)  # Draw a red circle at the centroid
                                mg.save_tif(images, name=".tif", path=Soma_centroid_path, variable=centroid_img)
                                #plt.imshow(centroid_img)
                        
                        
                        print ("...")
                        print ("...")
                        print("Cell centroids added!!")
                        print ("...")
                        print ("...")
                        
                        
                        
                        # FORTH ANALYSIS: SHOLL ANALYSIS
                        # GENERATING THE SKELETON WITH THE SOMA TO ASSES THE CROSSING PROCESSES, ETC
      
                        
                        print ("...")
                        print ("...")
                        print("Generating THE IMAGES FOR SHOLL ANALYSIS...")
                        print ("...")
                        print ("...")
                        
                        
                        
                        for image in os.listdir(Skeleton_path):
                            if image.endswith(".tif"):
                                input_skeleton = os.path.join(Skeleton_path, image)
                                input_soma = os.path.join(Soma_centroid_path, image)
                                
                                
                                skeleton_img = cv2.imread(input_skeleton, cv2.IMREAD_COLOR)
                                soma_img = cv2.imread(input_soma, cv2.IMREAD_COLOR)
                                soma_img = cv2.cvtColor(soma_img, cv2.COLOR_BGR2RGB)
                                
    
                                
                                # Check if the images have the same dimensions
                                if skeleton_img.shape == soma_img.shape:
                                    cell_centroid_image = cv2.add(skeleton_img, soma_img)
                                    mg.save_tif(image, ".tif", Cell_centroid_path, cell_centroid_image)
                        
                        
                        
                        print ("...")
                        print ("...")
                        print("SHOLL ANALYSIS...")
                        print ("...")
                        print ("...")
                        
                        # SHOLL ANALYSIS
                        
                        df_sholl= pd.DataFrame(columns=["Cell", "Sholl_max_distance", "Sholl_crossing_processes", "Sholl_circles"])
                        
                        for image in os.listdir(Cell_centroid_path):
                            if image.endswith(".tif"):
                                input_sholl = os.path.join(Cell_centroid_path, image)
                                #input_soma = os.path.join(Soma_path, image)
                                
                                sholl_img = cv2.imread(input_sholl, cv2.IMREAD_COLOR)
                                #soma = cv2.imread(input_soma)
                                
                                
                                #sholl_img = cv2.imread(input_sholl, cv2.IMREAD_COLOR)
                                #sholl_image, sholl_max_distance, sholl_crossing_processes, circle_image = sholl_circles(sholl_img, soma)
                                sholl_image, sholl_max_distance, sholl_crossing_processes, circle_image = mg.sholl_circles(sholl_img)
    
                                circles = circle_image[:, :, 2]
                                sholl_num_circles = mg.count(circles)
    
                                mg.save_tif(image, ".tif", Sholl_path, sholl_image)
                                
                                
                                
                                
                                df_sholl.loc[len(df_sholl)] = [image, sholl_max_distance, sholl_crossing_processes, sholl_num_circles]
                        
                        
                        df_sholl = pd.DataFrame(columns=["Cell", "Sholl_max_distance", "Sholl_crossing_processes", "Sholl_circles"])
    
                        for image in os.listdir(Cell_centroid_path):
                            if image.endswith(".tif"):
                                input_sholl = os.path.join(Cell_centroid_path, image)
                                sholl_img = cv2.imread(input_sholl, cv2.IMREAD_COLOR)
                                sholl_image, sholl_max_distance, sholl_crossing_processes, circle_image = mg.sholl_circles(sholl_img)
                        
                                circles = circle_image[:, :, 2]
                                sholl_num_circles = mg.count(circles)
                        
                                mg.save_tif(image, ".tif", Sholl_path, sholl_image)
                        
                                df_sholl.loc[len(df_sholl)] = {
                                    "Cell": image,
                                    "Sholl_max_distance": sholl_max_distance,
                                    "Sholl_crossing_processes": sholl_crossing_processes,
                                    "Sholl_circles": sholl_num_circles
                                }
    
                        
                        df_sholl['Cell'] = df_sholl['Cell'].str.extract(r'(\d+)').astype(int)     
                        df_sholl = df_sholl.sort_values(by='Cell')   
                        df_sholl.to_csv(csv_path+"Sholl_Analysis.csv", index=False) 
                        
                        
                        print ("...")
                        print ("...")
                        print("SHOLL ANALYSIS DONE!")
                        print ("...")
                        print ("...")
    
    
    
    ################################################# SHOLL ANALYSIS ##############################################
    ###############################################################################################################
    ###############################################################################################################
    ############################################################################################################### 
    
                        
         
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    ######################################## CONVEX HULL: FRACTAL ANALYSIS ########################################
                       
                        
                        # FIFTH ANALYSIS: FRACTAL ANALYSIS
                        #POLYGON FROM THE THRESHOLDED CELL
    
                        print ("...")
                        print ("...")
                        print("Generating CONVEX HULL...")
                        print ("...")
                        print ("...")
                        
                                
                        df_Convex_Hull = pd.DataFrame(columns=["Cell", "Convex_Hull_area", "Convex_Hull_perimeter", "Convex_Hull_compactness", "Fractal_dimension"])
    
                        for image in os.listdir(Cells_thresh_path):
                            if image.endswith(".tif"):
                                input_file = os.path.join(Cells_thresh_path, image)
                                img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
                                
                                Fractal_dimension = mg.fractal_dimension(img)
                                
                        
                                contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                                # Create a polygon covering the entire cell including ramifications
                                convex_hull_image = mg.polygon(img)
                        
                                # Calculate the convex hull area using countNonZero
                                convex_hull_area = cv2.countNonZero(convex_hull_image)
                        
                                # Calculate the convex hull perimeter using the findContours method
                                contours, _ = cv2.findContours(convex_hull_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                convex_hull_perimeter = cv2.arcLength(contours[0], True)
                        
                                # Calculate the convex hull compactness
                                convex_hull_compactness = convex_hull_area / convex_hull_perimeter
                        
                                # Save the resulting polygon image with the same filename
                                mg.save_tif(image, ".tif", Convex_Hull_path, convex_hull_image)
                        
                                # Append the results to the DataFrame
                                df_Convex_Hull.loc[len(df_Convex_Hull)] = {
                                    "Cell": image,
                                    "Convex_Hull_area": convex_hull_area,
                                    "Convex_Hull_perimeter": convex_hull_perimeter,
                                    "Convex_Hull_compactness": convex_hull_compactness,
                                    "Fractal_dimension": Fractal_dimension
                                }
    
    
     
                        
    
                        
                        
                        print ("...")
                        print ("...")
                        print("CONVEX HULL extracted")
                        print ("...")
                        print ("...")
                        
                        print ("...")
                        print ("...")
                        print("Getting CONVEX HULL centroids...")
                        print ("...")
                        print ("...")
    
    
    
    
                        df_Convex_Hull2 = pd.DataFrame(columns=["Cell", "Convex_Hull_eccentricity", "Convex_Hull_feret_diameter", "Convex_Hull_orientation"])
                        
                        for convex_hull in os.listdir(Convex_Hull_path):
                            if convex_hull.endswith(".tif"):
                                input_polygon = os.path.join(Convex_Hull_path, convex_hull)
                                polygon_img = io.imread(input_polygon)
                        
                                contours, _ = cv2.findContours(polygon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                                if len(contours) > 0:
                                    polygon_contour = max(contours, key=cv2.contourArea)  # Obtain the largest contour
                        
                                    # Calculate the eccentricity
                                    _, (major_axis, minor_axis), _ = cv2.fitEllipse(polygon_contour)
                                    convex_hull_eccentricity = major_axis / minor_axis
                        
                                    # Calculate the Feret diameter (maximum caliper)
                                    min_rect = cv2.minAreaRect(polygon_contour)
                                    convex_hull_feret_diameter = max(min_rect[1])  # Maximum of width and height
                        
                                    # Calculate the orientation (angle of the Feret diameter)
                                    convex_hull_orientation = min_rect[2]  # Angle in degrees
                        
                                    # Append the results to the DataFrame
                                    df_Convex_Hull2.loc[len(df_Convex_Hull2)] = {
                                        "Cell": convex_hull,
                                        "Convex_Hull_eccentricity": convex_hull_eccentricity,
                                        "Convex_Hull_feret_diameter": convex_hull_feret_diameter,
                                        "Convex_Hull_orientation": convex_hull_orientation
                                    }
                        
                                    epsilon = 0.02 * cv2.arcLength(polygon_contour, True)
                                    vertices = cv2.approxPolyDP(polygon_contour, epsilon, True)
                        
                                    # Draw the centroid (red) and vertices (green) on the image
                                    convex_hull_cetroid_img = cv2.cvtColor(polygon_img, cv2.COLOR_GRAY2BGR)
                                    M = cv2.moments(polygon_contour)
                                    centroid_x = int(M['m10'] / M['m00'])
                                    centroid_y = int(M['m01'] / M['m00'])
                                    centroid = (centroid_x, centroid_y)
                                    cv2.circle(convex_hull_cetroid_img, centroid, 3, (0, 0, 255), -1)  # Red circle for centroid
                                    for vertex in vertices:
                                        x, y = vertex[0]
                                        cv2.circle(convex_hull_cetroid_img, (x, y), 2, (0, 255, 0), -1)  # Green circle for vertices
                        
                                    # Save the image with visualizations
                                    mg.save_tif(convex_hull, ".tif", Convex_Hull_Centroid_path, convex_hull_cetroid_img)
                        
                        
                        
                        # Merge the two DataFrames on the "Cell" column
                        df_Convex_Hull = pd.merge(df_Convex_Hull, df_Convex_Hull2, on='Cell', how='inner')
                        
                        # Extract numerical part from the "Cell" column
                        df_Convex_Hull['Cell'] = df_Convex_Hull['Cell'].str.extract(r'(\d+)').astype(int)
                        
                        df_Convex_Hull.to_csv(csv_path+"Convex_Hull_Analysis.csv", index=False) 
    
    
                        print("...")
                        print("...")
                        print("Convex_Hull centroids added!")
                        print("...")
                        print("...")
    
                        
         
            
         
    ######################################## CONVEX HULL: FRACTAL ANALYSIS ########################################
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################                    
       
    
    
    
    
    
    
    
    
    
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    
    
    
    
    
    
    ###############################################################################################################
    ###############################################################################################################
    ###############################################################################################################
    ################################################# DATAFRAME GENERAL #####################################
    
    
    
                        print ("...")
                        print ("...")
                        print("Generating DATAFRAME WITH ALL THE FEATURES ...")
                        print ("...")
                        print ("...")
                        
                        
                        
                        ####GENERANDO EL DATAFRAME UNIDO POR LAS CARACTERISTICAS DE LAS RAMIFICACIONES: 
                        df_positions['Cell'] = df_positions['Cell'].astype(int)
                        df_cell['Cell'] = df_cell['Cell'].astype(int)
                        df_area['Cell'] = df_area['Cell'].astype(int)
                        df_skeleton['Cell'] = df_skeleton['Cell'].astype(int)
                        df_Convex_Hull['Cell'] = df_Convex_Hull['Cell'].astype(int)
                        df_sholl['Cell'] = df_sholl['Cell'].astype(int)
    
                        merged_df = df_positions.merge(df_area, on="Cell", how="inner").merge(df_skeleton, on="Cell", how="inner").merge(df_Convex_Hull, on="Cell", how="inner").merge(df_cell, on="Cell", how="inner").merge(df_sholl, on="Cell", how="inner")
                        
                        
                        
                        
                        # Getting new features: solidity and convexity
                        merged_df['Cell_solidity'] = merged_df['Cell_area'] / merged_df['Convex_Hull_area']
                        merged_df['Cell_convexity'] = merged_df['Cell_perimeter'] / merged_df['Convex_Hull_perimeter']
    
    
    
                        #MODIFICAR ACORDE A TU SUJETO. ESTAS COLUMNAS SE AGREGAN MANUALMENTE AL DATAFRAME
                        #########    SUJETO, GRUPO, REGION ANALIZADA, LADO, CORTE 
                        
                        ############################################## Add columns 
                        
                        
                        # Sort the DataFrame by the "Cell" column as numeric values
                        merged_df = merged_df.sort_values(by='Cell', key=lambda x: x.astype(int))
                        
                        # Reset the index after sorting if needed
                        merged_df = merged_df.reset_index(drop=True)
                        
                        #THIS SECTION DEPENDS ON THE REGION AND SUBJECTS THAT YOU ARE ANALYZING
                        # THESE FEATURES COULD BE EXTRACTED FROM THE FILE BUT TO AVOID MANIPULATING
                        # THE NAME OF THE FILES, THESE COLUMNS ARE MANUALLY CREATED
                        
    
                        
                        #GENERAL ID 
                        ID= f"{subject}_{group}_{treatment}_{tissue}_{region}"
                        
                        # COLUMNS TO IDENTIFY THE SUBJECT
                        merged_df.insert(0, 'subject', subject)
                        merged_df.insert(1, 'group', group)
                        merged_df.insert(2, 'treatment', treatment)
                        merged_df.insert(3, 'tissue', tissue)
                        merged_df.insert(4, 'region', region)
                        
    
                        merged_df['ID'] = (merged_df['subject'] + "_" + merged_df['group'] + "_" +
                                           merged_df['treatment'] + "_" + merged_df['tissue'] + "_" +
                                           merged_df['region'])
                        
                        
                        merged_df = merged_df[ ['ID'] + [col for col in merged_df.columns if col != 'ID'] ]
    
                        
                        #SAVE DATAFRAME
                        csv_name = csv_path + "Cell_Morphology.csv"
                        merged_df.to_csv(csv_name, index=False)
    
    
                        print ("...")
                        print ("...")
                        print(f"DATAFRAME SAVED AS CSV AT: {csv_name}")
                        print ("...")
                        print ("...")
                        



# VARIABLES
regions = ["HILUS", "CA1"]
subjects = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
groups = ["CNEURO-10", "VEH", "CNEURO-01"]
treatments = ["SCO", "SS"]
tissues = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]

# Empty DataFrame to store merged data
merged_data = pd.DataFrame()

# Loop to concatenate all the individual DF of the samples to create one DF 
# which contains the information of all of them 
for region in regions:
    for subject in subjects:
        for group in groups:
            for treatment in treatments:
                for tissue in tissues:
                    # Look in each directory for the file "Cell_Morphology.csv"
                    data_directory = f"{subject}_{group}_{treatment}_{tissue}_{region}_FOTO1_PROCESSED/Data/"
                    data_file = "Cell_Morphology.csv"
                    data_path = os.path.join(o_path, data_directory, data_file)
                    
                    if os.path.isfile(data_path):
                        # Read each CSV file and merge them into a single DataFrame
                        df = pd.read_csv(data_path)
                        merged_data = pd.concat([merged_data, df], ignore_index=True)

# Creating new columns to identify the file origin of the cells and its category
merged_data['Cell_ID'] = ("C" + merged_data['Cell'].astype(str) + "_" +
                          merged_data['subject']+ "_" + 
                          merged_data['group'] + "_" + 
                          merged_data['treatment'] + "_" +
                          merged_data['tissue'] + "_" + 
                          merged_data['region'] )

merged_data['categories'] = (merged_data['group'] + "_" +
                             merged_data['treatment'] + "_" + 
                             merged_data['region'])

# Reorder the DataFrame columns to place 'Cell_ID' and 'categories' at the beginning
merged_data = merged_data[['Cell_ID'] + [col for col in merged_data.columns if col != 'Cell_ID']]
merged_data = merged_data[['categories'] + [col for col in merged_data.columns if col != 'categories']]

# Save to a CSV file as Morphology but in a different directory called "Merged_Data". 
# This new Morphology file contains the information of all the micrographs
merged_csv_path = os.path.join(merge_csv_path, "Morphology.csv")
merged_data.to_csv(merged_csv_path, index=False)
print(f"Merged CSV saved at: {merged_csv_path}")


# Record end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

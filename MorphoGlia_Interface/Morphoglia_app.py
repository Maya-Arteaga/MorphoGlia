#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 01:26:10 2024

@author: juanpablomaya
"""

import matplotlib
matplotlib.use('Agg') 
import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog, PhotoImage
import threading
import os
import sys
import time
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hdbscan
import umap.plot
import matplotlib.cm as cm
import cv2
import tifffile as tiff
from skimage.morphology import skeletonize
from skimage import io
import morphoglia as mg
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import shutil




#####################################################################################
####################################  FUNCTIONS  ####################################
#####################################################################################

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def set_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Function to return to the main window after processing
def return_to_main_window(processing_window):
    processing_window.destroy()
    create_main_window()
    
    
def show_completion_window(previous_window, returncode):
    previous_window.destroy()
    window = tk.Toplevel()
    window.title("Process Completed")
    #window.geometry("600x300")
    center_window(window, 600, 300) 

    if returncode == 0:
        title = "Success"
        message = "The process completed successfully!"
    else:
        title = "Error"
        message = "An error occurred during the process."

    label = tk.Label(window, text=title, font=("Helvetica", 22, "bold"))
    label.pack(pady=20)

    text_widget = tk.Label(window, text=message, wraplength=550, justify="left", font=("Helvetica", 16))
    text_widget.pack(pady=10)

    window.update_idletasks()
    window.minsize(text_widget.winfo_width() + 40, text_widget.winfo_height() + 150)
    create_main_window()


#####################################################################################
####################################  FUNCTIONS  ####################################
#####################################################################################





#####################################################################################
#################################### MORPHOLOGY ####################################
#####################################################################################

def morphology_nomenclature_window(previous_window):
    previous_window.destroy()
    window = tk.Tk()
    window.title("Nomenclature")
    center_window(window, 800, 700)

    # Display the main instruction label
    label = tk.Label(window, text="Please enter the nomenclature of your images", font=("Helvetica", 24, "bold"))
    label.pack(pady=15)
    
    # Provide an example to guide the user
    label = tk.Label(window, text=(
        'For example, if your folder contains images named "R1_VEH_SS_T1_CA1.tif", "R2_VEH_SCOP_T2_HILUS.tif", etc., you should enter the following:\n\n'
        
        '- Subjects: R1, R2, ..., Rn\n'
        '- Groups: VEH\n'
        '- Treatments: SS, SCOP\n'
        '- Tissues: T1, T2, ..., Tn\n'
        '- Regions: CA1, HILUS.\n'
    ), wraplength=750, justify="left", font=("Helvetica", 17))
    label.pack(pady=5)
    
    # Highlight important details regarding input format
    label = tk.Label(window, text=(
        'Please separate each entry with a comma (",").\n'
        'Ensure that there are no misspellings and that you use the correct uppercase and lowercase letters.'
    ), wraplength=750, justify="center", font=("Helvetica", 18, "bold"))
    label.pack(pady=2)


    def create_entry(label_text):
        frame = tk.Frame(window)
        frame.pack(pady=10)
        tk.Label(frame, text=label_text, font=("Helvetica", 22)).pack(side=tk.LEFT)
        entry = tk.Entry(frame, font=("Helvetica", 22))
        entry.pack(side=tk.LEFT)
        return entry

    subjects_entry = create_entry("Subjects:")
    groups_entry = create_entry("Groups:")
    treatments_entry = create_entry("Treatments:")
    tissues_entry = create_entry("Tissues:")
    regions_entry = create_entry("Regions:")

    def process_entries():
        user_input = {
            "regions": [region.strip() for region in regions_entry.get().split(',')],
            "subjects": [subject.strip() for subject in subjects_entry.get().split(',')],
            "groups": [group.strip() for group in groups_entry.get().split(',')],
            "treatments": [treatment.strip() for treatment in treatments_entry.get().split(',')],
            "tissues": [tissue.strip() for tissue in tissues_entry.get().split(',')]
        }
        directory_selection_window(window, user_input)

    button = tk.Button(window, text="Next", command=process_entries, font=("Helvetica", 20))
    button.pack(pady=20)

    def on_closing():
        window.destroy()
        create_main_window()  # Reopen the main window when this window is closed

    window.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the window close event

    window.update_idletasks()
    window.minsize(700, 400)
    window.mainloop()

def directory_selection_window(previous_window, user_input):
    previous_window.destroy()
    window = tk.Tk()
    window.title("Select Directory")
    center_window(window, 800, 500)

    # Instruction label for selecting the directory
    label = tk.Label(window, text=(
        '\n\n\nPlease select the directory containing the set of images for the morphology analysis.'
        '\n\nThe results of the analysis will be saved in this directory, within a folder named "MorphoGlia".\n\n'
    ), wraplength=750, justify="left", font=("Helvetica", 22))
    label.pack(pady=15)
    
    # Instruction regarding image preprocessing
    instructions = (
        "Ensure that the microglia cells are preprocessed as binary images, "
        "with the cells displayed in BLACK and the background in WHITE.\n\n"
    )
    label = tk.Label(window, text=instructions, wraplength=750, justify="center", font=("Helvetica", 22, "bold"))
    label.pack(pady=15)



    def select_directory():
        directory = filedialog.askdirectory()
        if directory:
            processing_morphology_analysis_window(window, directory, user_input)


    button = tk.Button(window, text="Select Directory", command=select_directory, font=("Helvetica", 16))
    button.pack(pady=20)
    
    def on_closing():
        window.destroy()
        create_main_window()  # Reopen the main window when this window is closed

    window.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the window close event

    window.update_idletasks()
    window.minsize(label.winfo_width() + 40, label.winfo_height() + 150)

    window.mainloop()

def processing_morphology_analysis_window(previous_window, directory, user_input):
    previous_window.destroy()
    window = tk.Toplevel()  # Use Toplevel instead of Tk for consistency
    window.title("Processing...")
    center_window(window, 800, 600)

    label = tk.Label(window, text="Processing your data...", font=("Helvetica", 24, "bold"))
    label.pack(pady=20)

    log_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=20, font=("Helvetica", 14))
    log_text.pack(pady=20, padx=20, expand=True, fill=tk.BOTH)

    def log_callback(message):
        log_text.insert(tk.END, message + '\n')
        log_text.see(tk.END)

    def on_completion(returncode):
        if returncode == 0:
            label.config(text="Process Completed Successfully!")
            window.after(3000, show_completion_window)  # Wait 7 seconds before proceeding
        else:
            label.config(text="Process Encountered Errors.")
            messagebox.showerror("Error", "An error occurred during processing. Please check the log.")
            window.after(10000, show_completion_window)  # Wait 7 seconds before proceeding

    def show_completion_window():
        window.destroy()
        completion_window()

    def update_label():
        while processing:
            for suffix in ['', '.', '..', '...']:
                label.config(text=f"Processing{suffix}")
                time.sleep(0.5)
                if not processing:
                    break

    def run_process():
        try:
            # Call the actual image processing function
            process_images(directory, user_input, log_callback)
            on_completion(0)  # Call on_completion with success code
        except Exception as e:
            log_callback(f"Error during processing: {str(e)}")
            on_completion(1)  # Call on_completion with error code

    global processing
    processing = True

    # Start the processing thread
    thread = threading.Thread(target=run_process)
    thread.start()

    # Start the label update (animation) thread
    animation_thread = threading.Thread(target=update_label)
    animation_thread.start()

    window.mainloop()



def completion_window():
    window = tk.Tk()
    window.title("Process Completed")
    center_window(window, 600, 300)

    title = "Images Processed Successfully"
    message = '\nYou can find your results in the "MorphoGlia" folder\n located within the directory you selected.'

    label = tk.Label(window, text=title, font=("Helvetica", 24, "bold"))
    label.pack(pady=20)

    text_widget = tk.Label(window, text=message, wraplength=750, justify="center", font=("Helvetica", 20))
    text_widget.pack(pady=10)

    def return_to_main():
        window.destroy()
        create_main_window()

    #return_button = tk.Button(window, text="Return to Main Window", command=return_to_main, font=("Helvetica", 16))
    #return_button.pack(pady=20)

    
    def on_closing():
        window.destroy()
        create_main_window()  # Reopen the main window when this window is closed

    window.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the window close event

    window.update_idletasks()
    window.minsize(label.winfo_width() + 40, label.winfo_height() + 150)

    window.mainloop()

def process_images(directory, user_input, log_callback):
    try:
        start_time = time.time()

        # Set and create paths
        log_callback(f"Setting paths based on input directory: {directory}")
        i_path = directory
        o_path = mg.set_path(i_path + "/MorphoGlia/")
        log_callback(f"Output path set to: {o_path}")
        ID_path = mg.set_path(o_path + "ID/")
        log_callback(f"ID path set to: {ID_path}")

        regions = user_input["regions"]
        subjects = user_input["subjects"]
        groups = user_input["groups"]
        treatments = user_input["treatments"]
        tissues = user_input["tissues"]

        # List of supported file extensions
        supported_extensions = ['.tif', '.tiff', '.png', '.jpg']

        for region in regions:
            for subject in subjects:
                for group in groups:
                    for treatment in treatments:
                        for tissue in tissues:
                            img_base = f"{subject}_{group}_{treatment}_{tissue}_{region}"
                            image_path = None

                            # Check each extension for the existence of the image file
                            for ext in supported_extensions:
                                potential_path = os.path.join(i_path, img_base + ext)
                                log_callback(f"Looking for image: {potential_path}")

                                if os.path.isfile(potential_path):
                                    image_path = potential_path
                                    log_callback(f"Found image: {image_path}")
                                    break  # Stop searching once the image is found

                            if image_path:
                                try:
                                    log_callback(f"Processing image: {image_path}")

                                    individual_img_path = mg.set_path(os.path.join(o_path, f"{subject}_{group}_{treatment}_{tissue}_{region}/"))
                                    Cells_path = mg.set_path(os.path.join(individual_img_path, "Cells/"))
                                    Cells_thresh_path = mg.set_path(os.path.join(Cells_path, "Cells_thresh/"))
                                    Soma_path = mg.set_path(os.path.join(Cells_path, "Soma/"))
                                    Skeleton_path = mg.set_path(os.path.join(Cells_path, "Skeleton/"))
                                    Branches_path = mg.set_path(os.path.join(Cells_path, "Branches/"))
                                    Skeleton2_path = mg.set_path(os.path.join(Cells_path, "Skeleton2/"))
                                    Branches2_path = mg.set_path(os.path.join(Cells_path, "Branches2/"))
                                    Skeleton_Soma_path = mg.set_path(os.path.join(Cells_path, "Skeleton_Soma/"))
                                    Soma_centroid_path = mg.set_path(os.path.join(Cells_path, "Soma_Centroids/"))
                                    Convex_Hull_path = mg.set_path(os.path.join(Cells_path, "Convex_Hull/"))
                                    Convex_Hull_Centroid_path = mg.set_path(os.path.join(Cells_path, "Convex_Hull_Centroid/"))
                                    Branches3_path = mg.set_path(os.path.join(Cells_path, "Branches3/"))
                                    Cell_centroid_path = mg.set_path(os.path.join(Cells_path, "Cell_centroid/"))
                                    Sholl_path = mg.set_path(os.path.join(Cells_path, "Sholl/"))
                                    csv_path = mg.set_path(os.path.join(individual_img_path, "Data/"))

                                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                                    

                                    image = cv2.bitwise_not(image)
                                        
                                    thresh = image
                                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
                                    min_area = 1000
                                    max_area = 24000000
                                    num_cells_filtered = 0
                                    cell_positions = []
                                    cell_num = []
                                    individual_cell = 0

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
                                            object_img = image[bounding_box_y:bounding_box_y + bounding_box_height, bounding_box_x:bounding_box_x + bounding_box_width]
                                            individual_cell += 1
                                            cell_num.append(individual_cell)
                                            output_filename = f"{img_base}_cell_{individual_cell}.tif"
                                            output_path = os.path.join(Cells_thresh_path, output_filename)
                                            tiff.imwrite(output_path, object_img)

                                    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                                    labeled_cell2 = 0
                                    for i in range(1, num_labels):
                                        area = stats[i, cv2.CC_STAT_AREA]
                                        if area < min_area or area > max_area:
                                            labels[labels == i] = 0
                                        else:
                                            bounding_box_x = stats[i, cv2.CC_STAT_LEFT]
                                            bounding_box_y = stats[i, cv2.CC_STAT_TOP]
                                            bounding_box_width = stats[i, cv2.CC_STAT_WIDTH]
                                            bounding_box_height = stats[i, cv2.CC_STAT_HEIGHT]
                                            cv2.rectangle(color_image, (bounding_box_x, bounding_box_y), (bounding_box_x + bounding_box_width, bounding_box_y + bounding_box_height), (255, 255, 0), 2)
                                            labeled_cell2 += 1
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            bottom_left = (bounding_box_x, bounding_box_y + bounding_box_height + 20)
                                            font_scale = 0.5
                                            color = (255, 0, 0)
                                            thickness = 2
                                            cv2.putText(color_image, str(labeled_cell2), bottom_left, font, font_scale, color, thickness)
                                    mg.save_tif(output_filename, name=".tif", path=ID_path, variable=color_image)

                                    mg.name_to_number(Cells_thresh_path)
                                    df_positions = pd.DataFrame({'Cell': cell_num, 'cell_positions': cell_positions})
                                    df_positions = df_positions.sort_values(by='Cell', key=lambda x: x.astype(int))
                                    df_positions.to_csv(csv_path + "Cell_Positions.csv", index=False)

                                    for images in os.listdir(Cells_thresh_path):
                                        if images.endswith(".tif"):
                                            input_file = os.path.join(Cells_thresh_path, images)
                                            image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
                                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
                                            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                                            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                                            result = cv2.bitwise_and(image, image, mask=mask)
                                            mg.save_tif(images, name=".tif", path=Cells_thresh_path, variable=result)

                                    cell_data = []
                                    for cell in os.listdir(Cells_thresh_path):
                                        if cell.endswith(".tif"):
                                            thresh_cell_path = os.path.join(Cells_thresh_path, cell)
                                            Cell_area, Cell_perimeter, Cell_circularity, Cell_compactness, Cell_orientation, Cell_feret_diameter, Cell_eccentricity, Cell_aspect_ratio = mg.cell_analysis(thresh_cell_path)
                                            cell_data.append({
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

                                    df_cell = pd.DataFrame(cell_data)
                                    df_cell['Cell'] = df_cell['Cell'].str.extract(r'(\d+)').astype(int)
                                    df_cell = df_cell.sort_values(by='Cell')
                                    df_cell.to_csv(csv_path + "Cell_features.csv", index=False)

                                    for images in os.listdir(Cells_thresh_path):
                                        if images.endswith(".tif"):
                                            input_file = os.path.join(Cells_thresh_path, images)
                                            image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
                                            kernel = np.ones((3, 3), np.uint8)
                                            eroded = cv2.erode(image, kernel, iterations=4)
                                            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
                                            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                                            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                                            result = cv2.bitwise_and(eroded, eroded, mask=mask)
                                            dilated = cv2.dilate(result, kernel, iterations=4)
                                            height, width = dilated.shape[:2]
                                            frame_thickness = 10
                                            mask = np.ones(dilated.shape[:2], dtype="uint8") * 255
                                            mask[:frame_thickness, :] = 0
                                            mask[height - frame_thickness:, :] = 0
                                            mask[:, :frame_thickness] = 0
                                            mask[:, width - frame_thickness:] = 0
                                            result = cv2.bitwise_and(dilated, dilated, mask=mask)
                                            mg.save_tif(images, name=".tif", path=Soma_path, variable=result)

                                    mg.name_to_number(Soma_path)
                                    soma_data = []
                                    for soma in os.listdir(Soma_path):
                                        if soma.endswith(".tif"):
                                            cell_soma_path = os.path.join(Soma_path, soma)
                                            Soma_area, Soma_perimeter, Soma_circularity, Soma_compactness, Soma_orientation, Soma_feret_diameter, Soma_eccentricity, Soma_aspect_ratio = mg.soma_analysis(cell_soma_path)
                                            soma_data.append({
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

                                    df_area = pd.DataFrame(soma_data)
                                    df_area['Cell'] = df_area['Cell'].str.extract(r'(\d+)').astype(int)
                                    df_area = df_area.sort_values(by='Cell')
                                    df_area.to_csv(csv_path + "Soma_features.csv", index=False)

                                    for cell in os.listdir(Cells_thresh_path):
                                        if cell.endswith(".tif"):
                                            input_cell = os.path.join(Cells_thresh_path, cell)
                                            cell_img = cv2.imread(input_cell, cv2.IMREAD_GRAYSCALE)
                                            scale = cell_img / 255
                                            skeleton = skeletonize(scale)
                                            clean_skeleton = mg.erase(skeleton, 40)
                                            mg.save_tif(cell, name=".tif", path=Skeleton_path, variable=clean_skeleton)
                                            M, colored_image = mg.detect_and_color(clean_skeleton)
                                            mg.save_tif(cell, name=".tif", path=Branches_path, variable=colored_image)

                                    mg.name_to_number(Branches_path)

                                    for image in os.listdir(Skeleton_path):
                                        if image.endswith(".tif"):
                                            input_skeleton = os.path.join(Skeleton_path, image)
                                            input_soma = os.path.join(Soma_path, image)
                                            skeleton_img = cv2.imread(input_skeleton, cv2.IMREAD_GRAYSCALE)
                                            soma_img = cv2.imread(input_soma, cv2.IMREAD_GRAYSCALE)
                                            if skeleton_img.shape == soma_img.shape:
                                                subtracted_image = cv2.subtract(skeleton_img, soma_img)
                                                mg.save_tif(image, ".tif", Skeleton2_path, subtracted_image)
                                                M, colored_image = mg.detect_and_color(subtracted_image)
                                                mg.save_tif(image, name=".tif", path=Branches2_path, variable=colored_image)
                                                added_image = cv2.add(skeleton_img, soma_img)
                                                mg.save_tif(image, ".tif", Skeleton_Soma_path, added_image)

                                    for image in os.listdir(Skeleton_Soma_path):
                                        if image.endswith(".tif"):
                                            input_skeleton = os.path.join(Skeleton_Soma_path, image)
                                            input_soma = os.path.join(Soma_path, image)
                                            skeleton_img = cv2.imread(input_skeleton, cv2.IMREAD_GRAYSCALE)
                                            soma_img = cv2.imread(input_soma, cv2.IMREAD_GRAYSCALE)
                                            if skeleton_img.shape == soma_img.shape:
                                                subtracted_image = cv2.subtract(skeleton_img, soma_img)
                                                M, colored_image3 = mg.detect_features(subtracted_image, soma_img)
                                                mg.save_tif(image, name=".tif", path=Branches3_path, variable=colored_image3)

                                    df_skeleton = pd.DataFrame(columns=["Cell", "End_Points", "Junctions", "Branches", "Initial_Points", "Total_Branches_Length"])
                                    for skeleton in os.listdir(Branches3_path):
                                        if skeleton.endswith(".tif"):
                                            skeleton_image_path = os.path.join(Branches3_path, skeleton)
                                            skeleton_img = io.imread(skeleton_image_path)
                                            End_points = skeleton_img[:, :, 0] == 255
                                            num_end_points = mg.count(End_points)
                                            Junction_points = skeleton_img[:, :, 1] == 255
                                            num_junction_points = mg.count(Junction_points)
                                            Length = skeleton_img[:, :, 2]
                                            branches_length = mg.count(Length) + num_end_points
                                            Branches = skeleton_img[:, :, 2]
                                            num_branches = mg.count_branches(Branches)
                                            Initial_points = skeleton_img[:,:,1] == 200
                                            num_initial_points = mg.count(Initial_points)
                                            df_skeleton.loc[len(df_skeleton)] = {
                                                "Cell": skeleton,
                                                "End_Points": num_end_points,
                                                "Junctions": num_junction_points,
                                                "Branches": num_branches,
                                                "Initial_Points": num_initial_points,
                                                "Total_Branches_Length": branches_length
                                            }

                                    df_skeleton['Cell'] = df_skeleton['Cell'].str.extract(r'(\d+)').astype(int)
                                    df_skeleton = df_skeleton.sort_values(by='Cell')
                                    df_skeleton['ratio_branches'] = df_skeleton['End_Points'] / df_skeleton['Initial_Points']
                                    df_skeleton.to_csv(csv_path + "Skeleton_features.csv", index=False)

                                    centroids_soma = []
                                    for images in os.listdir(Soma_path):
                                        if images.endswith(".tif"):
                                            input_file = os.path.join(Soma_path, images)
                                            img = io.imread(input_file)
                                            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                            if len(contours) > 0:
                                                largest_contour = max(contours, key=cv2.contourArea)
                                                M = cv2.moments(largest_contour)
                                                if M['m00'] != 0:
                                                    centroid_x = int(M['m10'] / M['m00'])
                                                    centroid_y = int(M['m01'] / M['m00'])
                                                    centroid = (centroid_x, centroid_y)
                                                    centroids_soma.append((centroid_x, centroid_y))
                                                    log_callback(f"Centroid coordinates: {centroid_x}, {centroid_y}")
                                                else:
                                                    log_callback("Object has no area (m00=0)")
                                            else:
                                                log_callback("No contours found in the image")
                                            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                                            centroid_img = cv2.circle(img_color, centroid, 1, (255, 0, 0), -1)
                                            mg.save_tif(images, name=".tif", path=Soma_centroid_path, variable=centroid_img)

                                    for image in os.listdir(Skeleton_path):
                                        if image.endswith(".tif"):
                                            input_skeleton = os.path.join(Skeleton_path, image)
                                            input_soma = os.path.join(Soma_centroid_path, image)
                                            skeleton_img = cv2.imread(input_skeleton, cv2.IMREAD_COLOR)
                                            soma_img = cv2.imread(input_soma, cv2.IMREAD_COLOR)
                                            soma_img = cv2.cvtColor(soma_img, cv2.COLOR_BGR2RGB)
                                            if skeleton_img.shape == soma_img.shape:
                                                cell_centroid_image = cv2.add(skeleton_img, soma_img)
                                                mg.save_tif(image, ".tif", Cell_centroid_path, cell_centroid_image)

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

                                    df_Convex_Hull = pd.DataFrame(columns=["Cell", "Convex_Hull_area", "Convex_Hull_perimeter", "Convex_Hull_compactness", "Fractal_dimension"])
                                    for image in os.listdir(Cells_thresh_path):
                                        if image.endswith(".tif"):
                                            input_file = os.path.join(Cells_thresh_path, image)
                                            img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
                                            Fractal_dimension = mg.fractal_dimension(img)
                                            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                            convex_hull_image = mg.polygon(img)
                                            convex_hull_area = cv2.countNonZero(convex_hull_image)
                                            contours, _ = cv2.findContours(convex_hull_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                            convex_hull_perimeter = cv2.arcLength(contours[0], True)
                                            convex_hull_compactness = convex_hull_area / convex_hull_perimeter
                                            mg.save_tif(image, ".tif", Convex_Hull_path, convex_hull_image)
                                            df_Convex_Hull.loc[len(df_Convex_Hull)] = {
                                                "Cell": image,
                                                "Convex_Hull_area": convex_hull_area,
                                                "Convex_Hull_perimeter": convex_hull_perimeter,
                                                "Convex_Hull_compactness": convex_hull_compactness,
                                                "Fractal_dimension": Fractal_dimension
                                            }

                                    df_Convex_Hull2 = pd.DataFrame(columns=["Cell", "Convex_Hull_eccentricity", "Convex_Hull_feret_diameter", "Convex_Hull_orientation"])
                                    for convex_hull in os.listdir(Convex_Hull_path):
                                        if convex_hull.endswith(".tif"):
                                            input_polygon = os.path.join(Convex_Hull_path, convex_hull)
                                            polygon_img = io.imread(input_polygon)
                                            contours, _ = cv2.findContours(polygon_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                            if len(contours) > 0:
                                                polygon_contour = max(contours, key=cv2.contourArea)
                                                _, (major_axis, minor_axis), _ = cv2.fitEllipse(polygon_contour)
                                                convex_hull_eccentricity = major_axis / minor_axis
                                                min_rect = cv2.minAreaRect(polygon_contour)
                                                convex_hull_feret_diameter = max(min_rect[1])
                                                convex_hull_orientation = min_rect[2]
                                                df_Convex_Hull2.loc[len(df_Convex_Hull2)] = {
                                                    "Cell": convex_hull,
                                                    "Convex_Hull_eccentricity": convex_hull_eccentricity,
                                                    "Convex_Hull_feret_diameter": convex_hull_feret_diameter,
                                                    "Convex_Hull_orientation": convex_hull_orientation
                                                }
                                                epsilon = 0.02 * cv2.arcLength(polygon_contour, True)
                                                vertices = cv2.approxPolyDP(polygon_contour, epsilon, True)
                                                convex_hull_cetroid_img = cv2.cvtColor(polygon_img, cv2.COLOR_GRAY2BGR)
                                                M = cv2.moments(polygon_contour)
                                                centroid_x = int(M['m10'] / M['m00'])
                                                centroid_y = int(M['m01'] / M['m00'])
                                                centroid = (centroid_x, centroid_y)
                                                cv2.circle(convex_hull_cetroid_img, centroid, 3, (0, 0, 255), -1)
                                                for vertex in vertices:
                                                    x, y = vertex[0]
                                                    cv2.circle(convex_hull_cetroid_img, (x, y), 2, (0, 255, 0), -1)
                                                mg.save_tif(convex_hull, ".tif", Convex_Hull_Centroid_path, convex_hull_cetroid_img)

                                    df_Convex_Hull = pd.merge(df_Convex_Hull, df_Convex_Hull2, on='Cell', how='inner')
                                    df_Convex_Hull['Cell'] = df_Convex_Hull['Cell'].str.extract(r'(\d+)').astype(int)
                                    df_Convex_Hull.to_csv(csv_path+"Convex_Hull_Analysis.csv", index=False)

                                    df_positions['Cell'] = df_positions['Cell'].astype(int)
                                    df_cell['Cell'] = df_cell['Cell'].astype(int)
                                    df_area['Cell'] = df_area['Cell'].astype(int)
                                    df_skeleton['Cell'] = df_skeleton['Cell'].astype(int)
                                    df_Convex_Hull['Cell'] = df_Convex_Hull['Cell'].astype(int)
                                    df_sholl['Cell'] = df_sholl['Cell'].astype(int)

                                    merged_df = df_positions.merge(df_area, on="Cell", how="inner").merge(df_skeleton, on="Cell", how="inner").merge(df_Convex_Hull, on="Cell", how="inner").merge(df_cell, on="Cell", how="inner").merge(df_sholl, on="Cell", how="inner")
                                    merged_df['Cell_solidity'] = merged_df['Cell_area'] / merged_df['Convex_Hull_area']
                                    merged_df['Cell_convexity'] = merged_df['Cell_perimeter'] / merged_df['Convex_Hull_perimeter']
                                    merged_df = merged_df.sort_values(by='Cell', key=lambda x: x.astype(int))
                                    merged_df = merged_df.reset_index(drop=True)
                                    ID = f"{subject}_{group}_{treatment}_{tissue}_{region}"
                                    merged_df.insert(0, 'subject', subject)
                                    merged_df.insert(1, 'group', group)
                                    merged_df.insert(2, 'treatment', treatment)
                                    merged_df.insert(3, 'tissue', tissue)
                                    merged_df.insert(4, 'region', region)
                                    merged_df['ID'] = (merged_df['subject'] + "_" + merged_df['group'] + "_" + merged_df['treatment'] + "_" + merged_df['tissue'] + "_" + merged_df['region'])
                                    merged_df = merged_df[ ['ID'] + [col for col in merged_df.columns if col != 'ID'] ]
                                    csv_name = csv_path + "Cell_Morphology.csv"
                                    merged_df.to_csv(csv_name, index=False)
                                    log_callback ("...")
                                    log_callback ("...")
                                    log_callback(f"DATAFRAME SAVED AS CSV AT: {csv_name}")
                                    log_callback ("...")
                                    log_callback ("...")
                                except Exception as e:
                                    log_callback(f"Error processing image {image_path}: {e}")
                            else:
                                log_callback(f"Image not found for combination: {subject}, {group}, {treatment}, {tissue}, {region}. Skipping to next combination.")
        
        # Empty DF
        merged_data = pd.DataFrame()

        # Loop to concatenate all the individual DF of the samples to create one DF
        # which contains the information of all of them
        for region in regions:
            for subject in subjects:
                for group in groups:
                    for treatment in treatments:
                        for tissue in tissues:
                            # Look in each directory the file "Cell_Morphology.csv"
                            data_directory = f"{subject}_{group}_{treatment}_{tissue}_{region}/Data/"
                            data_file = "Cell_Morphology.csv"
                            data_path = os.path.join(o_path, data_directory, data_file)

                            if os.path.isfile(data_path):
                                # Read each CSV file and merge them into a single DataFrame
                                df = pd.read_csv(data_path)
                                merged_data = pd.concat([merged_data, df])

        # Creating new columns to identify the file origin of the cells and its category
        merged_data['Cell_ID'] = ("C" + merged_data['Cell'].astype(str) + "_" +
                                  merged_data['region'] + "_" +
                                  merged_data['group'] + "_" +
                                  merged_data['treatment'] + "_" +
                                  merged_data['tissue'] + "_" +
                                  merged_data['subject'])

        merged_data['categories'] = (merged_data['group'] + "_" +
                                     merged_data['treatment'] + "_" +
                                     merged_data['region'])

        merged_data = merged_data[ ['Cell_ID'] + [col for col in merged_data.columns if col != 'Cell_ID'] ]
        merged_data = merged_data[ ['categories'] + [col for col in merged_data.columns if col != 'categories'] ]

        # Save to a CSV file
        # as Morphology but in a different directory called "Merged_Data".
        # This new Morphology file contains the information of all the micrographs
        merged_csv_path = os.path.join(o_path, "Data", "Morphology.csv")
        os.makedirs(os.path.dirname(merged_csv_path), exist_ok=True)
        merged_data.to_csv(merged_csv_path, index=False)
        log_callback(f"Data saved at: {merged_csv_path}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        log_callback(f"Elapsed Time: {elapsed_time} seconds")


    except Exception as e:
        log_callback(f"An error occurred: {e}")


def run_morphometrics_analysis_script(directory, user_input, log_callback, completion_callback):
    try:
        process_images(directory, user_input, log_callback)
        completion_callback(0)
    except Exception as e:
        log_callback(f"An error occurred: {e}")
        completion_callback(1)






#####################################################################################
#################################### MORPHOLOGY ####################################
#####################################################################################




#####################################################################################
####################################     RFE     ####################################
#####################################################################################




def run_RFE_script(script_name, test_size, directory, log_callback, completion_callback):
    try:
        RFE_run_feature_selection(test_size, directory, log_callback)
        completion_callback(0)
    except Exception as e:
        log_callback(f"An error occurred: {e}")
        completion_callback(1)

def start_rfe_analysis(test_size, directory):
    RFE_processing_window("Feature_Selection_RFE_RF.py", test_size, directory)

def RFE_processing_window(script_name, test_size, directory):
    window = tk.Toplevel()
    window.title("Processing")
    #window.geometry("600x300")
    center_window(window, 600, 300)
    

    label = tk.Label(window, text="Processing...", font=("Helvetica", 22, "bold"))
    label.pack(pady=10)

    log_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=10, font=("Helvetica", 12))
    log_text.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    def log_callback(message):
        log_text.insert(tk.END, message + '\n')
        log_text.see(tk.END)

    def on_completion(returncode):
        global processing
        processing = False
        window.after(5000, lambda: RFE_show_completion_window(window, returncode))

    def update_label():
        while processing:
            for suffix in ['', '.', '..', '...']:
                label.config(text=f"Processing{suffix}")
                time.sleep(0.5)
                if not processing:
                    break

    global processing
    processing = True
    thread = threading.Thread(target=run_RFE_script, args=(script_name, test_size, directory, log_callback, on_completion))
    thread.start()

    animation_thread = threading.Thread(target=update_label)
    animation_thread.start()

    window.mainloop()

def RFE_show_completion_window(previous_window, returncode):
    previous_window.destroy()
    window = tk.Toplevel()
    window.title("Process Completed")
    #window.geometry("600x300")
    center_window(window, 600, 300)

    if returncode == 0:
        title = "Feature Selection Completed Successfully"
        message = '\nYou can find your results in the'
        message_path= '"MorphoGlia > Data > Feature_Selection" folder'

    else:
        title = "Error"
        message = "An error occurred during the process."

    label = tk.Label(window, text=title, font=("Helvetica", 22, "bold"))
    label.pack(pady=20)

    text_widget = tk.Label(window, text=message, wraplength=550, justify="center", font=("Helvetica", 19))
    text_widget.pack(pady=5)
    
    text_widget = tk.Label(window, text=message_path, wraplength=550, justify="center", font=("Helvetica", 19, "bold"))
    text_widget.pack(pady=5)

    window.update_idletasks()
    window.minsize(text_widget.winfo_width() + 40, text_widget.winfo_height() + 150)

def RFE_get_test_size(directory):
    def submit():
        try:
            test_size = float(entry.get())
            if 0.0 <= test_size <= 1.0:
                input_window.destroy()
                start_rfe_analysis(test_size, directory)
            else:
                messagebox.showerror("Invalid input", "Please enter a float between 0.0 and 1.0.")
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter a valid float.")

    input_window = tk.Tk()
    input_window.title("Input")
    #input_window.geometry("300x150")
    center_window(input_window, 300, 150)

    label = tk.Label(input_window, text="Please enter the test size (0.0-1.0):", font=("Helvetica", 14))
    label.pack(pady=10)

    entry = tk.Entry(input_window, font=("Helvetica", 16))
    entry.pack(pady=5)

    submit_button = tk.Button(input_window, text="OK", command=submit, font=("Helvetica", 14))
    submit_button.pack(pady=10)

    input_window.mainloop()

def RFE_select_directory():
    directory = filedialog.askdirectory()
    if directory:
        RFE_get_test_size(directory)

def RFE_show_initial_message():
    initial_window = tk.Tk()
    initial_window.title("Information")
    #initial_window.geometry("600x300")
    center_window(initial_window, 700, 300)

    label = tk.Label(
        initial_window, 
        text=(
            '\nPlease select the directory containing the "Morphology.csv" file generated during the Morphology Analysis. '
            'This file should be located in "MorphoGlia > Data".\n\n'
            'The resulting plots will be saved in the same directory, under "Feature_Selection".'
        ),
        wraplength=550, 
        justify="left", 
        font=("Helvetica", 20)
    )
    label.pack(pady=20)

    def on_ok():
        initial_window.destroy()
        RFE_select_directory()

    button = tk.Button(initial_window, text="OK", command=on_ok, font=("Helvetica", 14))
    button.pack(pady=10)

    initial_window.mainloop()

def RFE_run_feature_selection(test_size, directory, log_callback):
    try:
        # Record start time
        start_time = time.time()

        # Load the data
        i_path = directory
        Plot_path = mg.set_path(i_path + "/Feature_Selection/")

        fraction_variables = 2
        data = pd.read_csv(i_path + "/Morphology.csv")

        # Specify the attributes you want to use in the decision tree
        selected_attributes = [
            'Soma_area', 'Soma_perimeter', 'Soma_circularity',
            'Soma_compactness', 'Soma_orientation', 'Soma_feret_diameter',
            'Soma_eccentricity', 'Soma_aspect_ratio', 'End_Points',
            'Junctions', 'Branches', 'Initial_Points', 'Total_Branches_Length',
            'ratio_branches', 'Convex_Hull_area', 'Convex_Hull_perimeter',
            'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Fractal_dimension',
            'Convex_Hull_feret_diameter', 'Cell_area', 'Cell_perimeter', 'Cell_circularity',
            'Cell_compactness', 'Cell_feret_diameter', 'Cell_eccentricity',
            'Cell_aspect_ratio', 'Sholl_max_distance', 'Sholl_crossing_processes',
            'Sholl_circles', 'Cell_solidity', 'Cell_convexity'
        ]

        y = data['categories']
        X = data[selected_attributes]

        # Correlation matrix for all features
        correlation_matrix = data[selected_attributes].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 5},
                    xticklabels=[label.replace('_', ' ') for label in selected_attributes],
                    yticklabels=[label.replace('_', ' ') for label in selected_attributes])
        plt.title('Correlation Matrix Heatmap for All Features', fontsize=16, fontweight='bold')
        plt.xticks(rotation=90, fontsize=8, fontweight='bold')
        plt.yticks(rotation=0, fontsize=8, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Plot_path + "Correlation_Matrix_All_Features.png", dpi=300, bbox_inches="tight")
        plt.close()  # Close the plot to free memory
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=24)
    
        rf = RandomForestClassifier(random_state=42)
        num_features_to_keep = X.shape[1] // fraction_variables
        rfe = RFE(estimator=rf, n_features_to_select=num_features_to_keep)
        rfe.fit(X_train, y_train)
    
        selected_features = [feature for feature, rank in zip(selected_attributes, rfe.ranking_) if rank == 1]
    
        X_train_selected = rfe.transform(X_train)
        X_test_selected = rfe.transform(X_test)
    
        rf.fit(X_train_selected, y_train)
    
        y_pred = rf.predict(X_test_selected)
    
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
    
        # Save selected features and their importance to a CSV file
        feature_importances = rf.feature_importances_
        importance_df = pd.DataFrame({'Selected Features': selected_features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        importance_df.to_csv(Plot_path + "Selected_Features_Importance.csv", index=False)
    
        # Save performance metrics to a CSV file
        performance_metrics = {
            'Metric': ['Accuracy', 'F1 Score'],
            'Value': [accuracy, f1]
        }
        performance_df = pd.DataFrame(performance_metrics)
        performance_df.to_csv(Plot_path + "Performance_Metrics.csv", index=False)
    
        # Save classification report to a CSV file
        classification_df = pd.DataFrame(report).transpose()
        classification_df.to_csv(Plot_path + "Classification_Report.csv", index=True)
    
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y),
                    yticklabels=np.unique(y))
        plt.title(f'Confusion Matrix for test_size={test_size}', fontweight='bold')
        plt.xlabel('Predicted', fontweight='bold')
        plt.ylabel('True', fontweight='bold')
        plt.xticks(rotation=0, fontweight='bold')
        plt.yticks(rotation=0, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Plot_path + f"Confusion_Matrix_RF_{test_size}_{fraction_variables}.png", dpi=300, bbox_inches="tight")
        plt.close()  # Close the plot to free memory
    
        # Replace underscores with spaces for display
        importance_df_display = importance_df.copy()
        importance_df_display['Selected Features'] = importance_df_display['Selected Features'].str.replace('_', ' ')
        
        # Reverse the Blues palette
        palette = sns.color_palette('Blues', len(importance_df_display))[::-1]
    
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Selected Features', data=importance_df_display, palette=palette)
        plt.xlabel('')
        plt.ylabel('Selected Features', fontweight='bold')
        plt.title(f'Feature Importances for test_size={test_size}', fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.tight_layout()
        plt.savefig(Plot_path + f"Feature_Importances_RF_{test_size}_{fraction_variables}.png", dpi=300, bbox_inches="tight")
        plt.close()  # Close the plot to free memory
    
        feature_RFE = importance_df['Selected Features'].tolist()
        data_RFE = data[feature_RFE]
    
        correlation_matrix = data_RFE.corr()
    
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8},
                    xticklabels=[label.replace('_', ' ') for label in feature_RFE],
                    yticklabels=[label.replace('_', ' ') for label in feature_RFE])
        plt.title(f'Correlation Matrix Heatmap for test_size={test_size}', fontsize=16, fontweight='bold')
        plt.xticks(rotation=90, fontsize=8, fontweight='bold')
        plt.yticks(rotation=0, fontsize=8, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Plot_path + f"Correlation_Matrix_RFE_{test_size}_{fraction_variables}.png", dpi=300, bbox_inches="tight")
        plt.close()  # Close the plot to free memory
        
        log_callback("________________________________________")
        log_callback("________________________________________")
        log_callback("")
        log_callback(f"Selected Features for test_size={test_size}: {selected_features}")
        log_callback("")
        log_callback("________________________________________")
        log_callback("________________________________________")
        log_callback("")
        log_callback(f"Accuracy on Test Set for test_size={test_size}: {accuracy}")
        log_callback("")
        log_callback("________________________________________")
        log_callback("________________________________________")
        log_callback("")
        log_callback(f"Classification Report for test_size={test_size}:\n{classification_report(y_test, y_pred)}")
        log_callback("")
        log_callback("________________________________________")
        log_callback("________________________________________")
    
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_min = elapsed_time / 60
    
        log_callback("")
        log_callback(f"Elapsed Time: {elapsed_time} seconds")
        log_callback(f"Elapsed Time: {elapsed_time_min} min")
        log_callback("")
        log_callback("________________________________________")
        log_callback("________________________________________")

    except Exception as e:
        log_callback(f"An unexpected error occurred: {e}")
        raise


#####################################################################################
####################################     RFE     ####################################
#####################################################################################





#####################################################################################
####################################     UMAP    ####################################
#####################################################################################



# GUI for UMAP Analysis
def processing_window_umap(directory):
    window = tk.Toplevel()
    window.title("Processing")
    #window.geometry("600x300")
    center_window(window, 600, 300) 
    

    label = tk.Label(window, text="Processing...", font=("Helvetica", 22, "bold"))
    label.pack(pady=10)

    log_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=10, font=("Helvetica", 12))
    log_text.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    def log_callback(message):
        log_text.insert(tk.END, message + '\n')
        log_text.see(tk.END)

    def on_completion(returncode):
        window.after(5000, lambda: UMAP_show_completion_window(window, returncode))

    def update_label():
        while processing:
            for suffix in ['', '.', '..', '...']:
                label.config(text=f"Processing{suffix}")
                time.sleep(0.5)
                if not processing:
                    break

    global processing
    processing = True
    thread = threading.Thread(target=run_umap_analysis, args=(directory, log_callback, on_completion))
    thread.start()

    animation_thread = threading.Thread(target=update_label)
    animation_thread.start()

    window.mainloop()
    
def UMAP_show_completion_window(previous_window, returncode):
    previous_window.destroy()
    window = tk.Toplevel()
    window.title("Process Completed")
    #window.geometry("600x300")
    center_window(window, 600, 300)
    
    

    if returncode == 0:
        title = "Dimensionality Reduction Completed Successfully"
        message = '\nYou can find your results in the'
        message_path= '"MorphoGlia > Data > UMAP" folder'

    else:
        title = "Error"
        message = "An error occurred during the process."

    label = tk.Label(window, text=title, font=("Helvetica", 22, "bold"))
    label.pack(pady=20)

    text_widget = tk.Label(window, text=message, wraplength=550, justify="center", font=("Helvetica", 19))
    text_widget.pack(pady=5)
    
    text_widget = tk.Label(window, text=message_path, wraplength=550, justify="center", font=("Helvetica", 19, "bold"))
    text_widget.pack(pady=5)

    window.update_idletasks()
    window.minsize(text_widget.winfo_width() + 40, text_widget.winfo_height() + 150)

def select_directory_umap():
    directory = filedialog.askdirectory()
    if directory:
        processing_window_umap(directory)
        
def UMAP_show_initial_message():
    initial_window = tk.Tk()
    initial_window.title("Information")
    #initial_window.geometry("600x300")
    center_window(initial_window, 700, 300)

    label = tk.Label(
        initial_window, 
        text=(
            '\nPlease select the directory containing the "Morphology.csv" file generated during the Morphology Analysis. '
            'This file should be located in "MorphoGlia > Data".\n\n'
            'The resulting plots will be saved in the same directory, under "UMAP".'
        ),
        wraplength=550, 
        justify="left", 
        font=("Helvetica", 20)
    )
    label.pack(pady=20)

    def on_ok():
        initial_window.destroy()
        select_directory_umap()

    button = tk.Button(initial_window, text="OK", command=on_ok, font=("Helvetica", 14))
    button.pack(pady=10)

    initial_window.mainloop()

def run_umap_analysis(directory, log_callback, completion_callback):
    try:
        start_time = time.time()
        UMAP_path = set_path(os.path.join(directory, "UMAP"))
        selected_features_path = os.path.join(directory, "Feature_Selection", "Selected_Features_Importance.csv")

        selected_features_df = pd.read_csv(selected_features_path)
        selected_features = selected_features_df['Selected Features'].tolist()

        data = pd.read_csv(os.path.join(directory, "Morphology.csv"))
        selected_data = data[selected_features]

        n_neighbors = [10, 15, 20, 30, 50]
        min_dist = [0.01, 0.05, 0.1]

        fig, axes = plt.subplots(nrows=len(n_neighbors), ncols=len(min_dist), figsize=(22, 25))
        for i, n in enumerate(n_neighbors):
            for j, d in enumerate(min_dist):
                reducer = umap.UMAP(n_neighbors=n, min_dist=d, random_state=24)
                embedding = reducer.fit_transform(selected_data)
                ax = axes[i, j]
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.4)
                ax.set_title(f'n_neighbors={n}, min_dist={d}', fontsize=14)
                ax.set_xlabel('UMAP 1', fontsize=10)
                ax.set_ylabel('UMAP 2', fontsize=10)
                ax.grid(False)

        plt.tight_layout()
        plt.savefig(os.path.join(UMAP_path, "UMAP_grid.png"), dpi=300)
        #plt.show()

        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_min = elapsed_time / 60

        log_callback(f"Elapsed Time: {elapsed_time} seconds\nElapsed Time: {elapsed_time_min} min")
        completion_callback(0)
    except Exception as e:
        log_callback(f"Error: {str(e)}")
        completion_callback(1)





#####################################################################################
####################################     UMAP    ####################################
#####################################################################################





#####################################################################################
####################################   HDBSCAN   ####################################
#####################################################################################

# GUI for HDBSCAN Analysis


def HDBSCAN_show_initial_message():
    initial_window = tk.Tk()
    initial_window.title("Information")
    #initial_window.geometry("600x300")
    center_window(initial_window, 700, 300)

    label = tk.Label(
        initial_window, 
        text=(
            '\nPlease select the directory containing the "Morphology.csv" file generated during the Morphology Analysis. '
            'This file should be located in "MorphoGlia > Data".\n\n'
            'The resulting plots will be saved in the same directory, under "UMAP_HDBSCAN".'
        ),
        wraplength=550, 
        justify="left", 
        font=("Helvetica", 20)
    )
    label.pack(pady=20)

    def on_ok():
        initial_window.destroy()
        select_directory_hdbscan()

    button = tk.Button(initial_window, text="OK", command=on_ok, font=("Helvetica", 14))
    button.pack(pady=10)

    initial_window.mainloop()




def HDBSCAN_get_parameters(directory):
    def submit():
        try:
            n_neighbors = int(n_neighbors_entry.get())
            min_dist = float(min_dist_entry.get())
            min_cluster_size = int(min_cluster_size_entry.get())
            min_samples = int(min_samples_entry.get())
            input_window.destroy()
            processing_window_hdbscan(directory, n_neighbors, min_dist, min_cluster_size, min_samples)
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid values for all parameters.")

    input_window = tk.Toplevel()
    input_window.title("Input Parameters")
    #input_window.geometry("400x400")
    center_window(input_window, 400, 400) 

    tk.Label(input_window, text="Number of neighbors:", font=("Helvetica", 14)).pack(pady=5)
    n_neighbors_entry = tk.Entry(input_window, font=("Helvetica", 14))
    n_neighbors_entry.pack(pady=5)

    tk.Label(input_window, text="Minimum distance:", font=("Helvetica", 14)).pack(pady=5)
    min_dist_entry = tk.Entry(input_window, font=("Helvetica", 14))
    min_dist_entry.pack(pady=5)

    tk.Label(input_window, text="Minimum cluster size:", font=("Helvetica", 14)).pack(pady=5)
    min_cluster_size_entry = tk.Entry(input_window, font=("Helvetica", 14))
    min_cluster_size_entry.pack(pady=5)

    tk.Label(input_window, text="Minimum samples:", font=("Helvetica", 14)).pack(pady=5)
    min_samples_entry = tk.Entry(input_window, font=("Helvetica", 14))
    min_samples_entry.pack(pady=5)

    tk.Button(input_window, text="Submit", command=submit, font=("Helvetica", 14)).pack(pady=10)

    input_window.mainloop()






def run_hdbscan_analysis(directory, n_neighbors, min_dist, min_cluster_size, min_samples, log_callback, completion_callback):
    try:
        start_time = time.time()
    
        # Load the data
        UMAP_HDBSCAN_path = set_path(os.path.join(directory, "UMAP_HDBSCAN"))
        selected_features_path = os.path.join(directory, "Feature_Selection", "Selected_Features_Importance.csv")
    
        selected_features_df = pd.read_csv(selected_features_path)
        selected_features = selected_features_df['Selected Features'].tolist()
    
        data = pd.read_csv(os.path.join(directory, "Morphology.csv"))
        selected_data = data[selected_features]
    
        # Apply UMAP for dimensionality reduction
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=24)
        embedding = reducer.fit_transform(selected_data)
    
        # Apply HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, allow_single_cluster=True)
        labels = clusterer.fit_predict(embedding)
    
        # Prepare colors for clusters
        color= 'rainbow'
        unique_labels = set(labels)
        colormap = cm.get_cmap(color, len(unique_labels))
        
        # Map colors for labels
        colors = [colormap(label / len(unique_labels)) if label != -1 else (0.0, 0.0, 0.0) for label in labels]  # Map -1 to black
        colors_rgb = [tuple(int(c * 255) for c in color) for color in colors]  # Convert RGB to 0-255 range
        
        # Convert colors to string format
        colors_str = [f"{int(color[0])},{int(color[1])},{int(color[2])}" for color in colors_rgb]
        
        # Save UMAP plot
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.4)
        plt.title(f'UMAP with n_neighbors={n_neighbors} and min_dist={min_dist}', fontsize=18)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.grid(False)
        plt.savefig(os.path.join(UMAP_HDBSCAN_path, f"UMAP_{n_neighbors}_{min_dist}.png"), dpi=300)
        plt.close()
    
        # Save HDBSCAN plot with legend outside
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=color, alpha=0.4)
        plt.title(f'HDBSCAN Clustering', fontsize=18)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.grid(False)
        
        # Create a legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colormap(i / len(unique_labels)), markersize=10, label=f'Cluster {i}') for i in unique_labels]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to make room for the legend
        plt.savefig(os.path.join(UMAP_HDBSCAN_path, f"UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.png"), dpi=300)
        plt.close()
    
        # Add columns to the original dataframe
        data['UMAP_1'] = embedding[:, 0]
        data['UMAP_2'] = embedding[:, 1]
        data['Clusters'] = labels
        data['Cluster_Color'] = colors_str
        

        
        # Save updated data
        data.to_csv(os.path.join(directory, f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"), index=False)
    
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_min = elapsed_time / 60

        log_callback(f"Elapsed Time: {elapsed_time} seconds\nElapsed Time: {elapsed_time_min} min")
        completion_callback(0)
    except Exception as e:
        log_callback(f"Error: {str(e)}")
        completion_callback(1)



def select_directory_hdbscan():
    directory = filedialog.askdirectory()
    if directory:
        HDBSCAN_get_parameters(directory)






def select_directory_hdbscan():
    directory = filedialog.askdirectory()
    if directory:
        HDBSCAN_get_parameters(directory)



def processing_window_hdbscan(directory, n_neighbors, min_dist, min_cluster_size, min_samples):
    window = tk.Toplevel()
    window.title("Processing")
    #window.geometry("600x300")
    center_window(window, 600, 300) 

    label = tk.Label(window, text="Processing...", font=("Helvetica", 22, "bold"))
    label.pack(pady=10)

    log_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=10, font=("Helvetica", 12))
    log_text.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    def log_callback(message):
        log_text.insert(tk.END, message + '\n')
        log_text.see(tk.END)

    def on_completion(returncode):
        global processing
        processing = False
        window.after(5000, lambda: HDBSCAN_show_completion_window(window, returncode))

    def update_label():
        while processing:
            for suffix in ['', '.', '..', '...']:
                label.config(text=f"Processing{suffix}")
                time.sleep(0.5)
                if not processing:
                    break

    global processing
    processing = True
    thread = threading.Thread(target=run_hdbscan_analysis, args=(directory, n_neighbors, min_dist, min_cluster_size, min_samples, log_callback, on_completion))
    thread.start()

    animation_thread = threading.Thread(target=update_label)
    animation_thread.start()

    window.mainloop()
    
    
    
    
def HDBSCAN_show_completion_window(previous_window, returncode):
    previous_window.destroy()
    window = tk.Toplevel()
    window.title("Process Completed")
    #window.geometry("600x300")
    center_window(window, 600, 300)
    
    

    if returncode == 0:
        title = "Clustering Completed Successfully"
        message = '\nYou can find your results in the'
        message_path= '"MorphoGlia > Data > UMAP_HDBSCAN" folder'

    else:
        title = "Error"
        message = "An error occurred during the process."

    label = tk.Label(window, text=title, font=("Helvetica", 22, "bold"))
    label.pack(pady=20)

    text_widget = tk.Label(window, text=message, wraplength=550, justify="center", font=("Helvetica", 19))
    text_widget.pack(pady=5)
    
    text_widget = tk.Label(window, text=message_path, wraplength=550, justify="center", font=("Helvetica", 19, "bold"))
    text_widget.pack(pady=5)

    window.update_idletasks()
    window.minsize(text_widget.winfo_width() + 40, text_widget.winfo_height() + 150)

#####################################################################################
####################################   HDBSCAN   ####################################
#####################################################################################






#####################################################################################
############################### SPATIAL VIZUALIZATION   #############################
#####################################################################################




def Spatial_vizualization_run_processing(directory, n_neighbors, min_dist, min_cluster_size, min_samples, regions, subjects, groups, treatments, tissues, log_callback, completion_callback):
    try:
        # Start processing
        i_path = directory
        o_path = os.path.join(i_path, "MorphoGlia/")
        csv_path = os.path.join(o_path, "Data")
        ID_path = os.path.join(o_path, "ID_clusters")
        os.makedirs(ID_path, exist_ok=True)  # Ensure ID_clusters directory exists

        csv_file = f"Morphology_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
        data = os.path.join(csv_path, csv_file)

        # Read the CSV file
        df = pd.read_csv(data)

        # Generate cluster colors from CSV
        cluster_colors = {}
        for _, row in df.iterrows():
            cluster = row['Clusters']
            color = tuple(map(int, row['Cluster_Color'].strip('()').split(',')))
            cluster_colors[cluster] = color

        # Iterate over categories and create separate CSV files if the group exists
        for region in regions:
            for s in subjects:
                for g in groups:
                    for tr in treatments:
                        for ti in tissues:
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
            for s in subjects:
                for g in groups:
                    for tr in treatments:
                        for ti in tissues:
                            csv_file_name = f"{s}_{g}_{tr}_{ti}_{region}_Morphology_UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
                            source_file = os.path.join(csv_path, csv_file_name)
                            target_directory = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_{region}", "Data")

                            if os.path.exists(source_file) and os.path.exists(target_directory):
                                shutil.copy2(source_file, target_directory)
                                os.remove(source_file)  # Remove the original file

        # Processing the images and generating the final composite image
        df_color_position = pd.DataFrame(columns=["Cell", "Clusters", "x1", "y1", "x2", "y2"])


        # List of supported file extensions
        supported_extensions = ['.tif', '.tiff', '.png', '.jpg']
        
        for region in regions:
            for s in subjects:
                for g in groups:
                    for tr in treatments:
                        for ti in tissues:
                            individual_img_path = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_{region}/")
                            csv_path = os.path.join(individual_img_path, "Data/")
                            csv_file = f"{s}_{g}_{tr}_{ti}_{region}_Morphology_UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
        
                            original_img = None
                            
                            # Search for the image file with any of the supported extensions
                            for ext in supported_extensions:
                                potential_img = os.path.join(individual_img_path, f"{s}_{g}_{tr}_{ti}_{region}{ext}")
                                if os.path.isfile(potential_img):
                                    original_img = potential_img
                                    break
        
                            if original_img:
                                # Proceed with processing the image
                                print(f"Found image: {original_img}")
                                # Add your image processing code here
                            else:
                                print(f"No image found for {s}_{g}_{tr}_{ti}_{region} with any of the supported extensions.")
                                    
                                    
                            
                            #########

                            if os.path.isfile(os.path.join(csv_path, csv_file)):
                                log_callback(f"Successfully loaded CSV file: {csv_file}")
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

        # Composite image generation
        
        
        supported_extensions = ['.tif', '.tiff', '.png', '.jpg']
        
        for region in regions:
            for s in subjects:
                for g in groups:
                    for tr in treatments:
                        for ti in tissues:
                            original_img = None
                            
                            # Search for the image file with any of the supported extensions
                            for ext in supported_extensions:
                                potential_img = f"{s}_{g}_{tr}_{ti}_{region}{ext}"
                                if os.path.isfile(os.path.join(i_path, potential_img)):
                                    original_img = potential_img
                                    break
                            
                            if original_img:
                                original_image_path = os.path.join(i_path, original_img)
                                original_image = cv2.imread(original_image_path)
                                log_callback(f"Successfully loaded image: {original_img}")
                                
                                if original_image is not None:
                                    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2RGBA)
                                    height, width, channels = original_image.shape
                                    empty_image = np.zeros((height, width, 4), np.uint8)
    
                                    individual_img_path = os.path.join(o_path, f"{s}_{g}_{tr}_{ti}_{region}/")
                                    csv_path = os.path.join(individual_img_path, "Data/")
                                    csv_file = f"{s}_{g}_{tr}_{ti}_{region}_Morphology_UMAP_HDBSCAN_{n_neighbors}_{min_dist}_{min_cluster_size}_{min_samples}.csv"
    
                                    if os.path.isfile(os.path.join(csv_path, csv_file)):
                                        data = pd.read_csv(os.path.join(csv_path, csv_file))
                                        
                                        Cells_path = os.path.join(individual_img_path, "Cells/")
                                        color_path = set_path(os.path.join(Cells_path, "Color_Cells/"))
    
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
                                        # Save the resulting image
                                        cluster_image_path = os.path.join(ID_path, f"{s}_{g}_{tr}_{ti}_{region}_Clusters.tif")
                                        cv2.imwrite(cluster_image_path, empty_image)
            completion_callback(0)
    except Exception as e:
        log_callback(f"Error: {str(e)}")
        completion_callback(1)
        




def Spatial_vizualization_nomenclature_window():
    window = tk.Tk()
    window.title("Nomenclature")
    center_window(window, 800, 700)

    # Display the main instruction label
    label = tk.Label(window, text="Please enter the nomenclature of your images", font=("Helvetica", 24, "bold"))
    label.pack(pady=15)
    
    # Provide an example to guide the user
    label = tk.Label(window, text=(
        'For example, if your folder contains images named "R1_VEH_SS_T1_CA1.tif", "R2_VEH_SCOP_T2_HILUS.tif", etc., you should enter the following:\n\n'
        
        '- Subjects: R1, R2, ..., Rn\n'
        '- Groups: VEH\n'
        '- Treatments: SS, SCOP\n'
        '- Tissues: T1, T2, ..., Tn\n'
        '- Regions: CA1, HILUS.\n'
    ), wraplength=750, justify="left", font=("Helvetica", 17))
    label.pack(pady=5)
    
    # Highlight important details regarding input format
    label = tk.Label(window, text=(
        'Please separate each entry with a comma (",").\n'
        'Ensure that there are no misspellings and that you use the correct uppercase and lowercase letters.'
    ), wraplength=750, justify="center", font=("Helvetica", 18, "bold"))
    label.pack(pady=2)


    def create_entry(label_text):
        frame = tk.Frame(window)
        frame.pack(pady=10)
        tk.Label(frame, text=label_text, font=("Helvetica", 22)).pack(side=tk.LEFT)
        entry = tk.Entry(frame, font=("Helvetica", 22))
        entry.pack(side=tk.LEFT)
        return entry

    subjects_entry = create_entry("Subjects:")
    groups_entry = create_entry("Groups:")
    treatments_entry = create_entry("Treatments:")
    tissues_entry = create_entry("Tissues:")
    regions_entry = create_entry("Regions:")

    def process_entries():
        user_input = {
            "regions": [region.strip() for region in regions_entry.get().split(',')],
            "subjects": [subject.strip() for subject in subjects_entry.get().split(',')],
            "groups": [group.strip() for group in groups_entry.get().split(',')],
            "treatments": [treatment.strip() for treatment in treatments_entry.get().split(',')],
            "tissues": [tissue.strip() for tissue in tissues_entry.get().split(',')]
        }
        window.destroy()
        Spatial_vizualization_select_directory(
            user_input["regions"], user_input["subjects"], user_input["groups"],
            user_input["treatments"], user_input["tissues"]
        )

    button = tk.Button(window, text="Next", command=process_entries, font=("Helvetica", 20))
    button.pack(pady=20)

    window.update_idletasks()
    window.minsize(700, 400)
    window.mainloop()

def Spatial_vizualization_select_directory(regions, subjects, groups, treatments, tissues):
    window = tk.Tk()
    window.title("Select Directory")
    center_window(window, 800, 300)

    label = tk.Label(window, text=(
        '\n\n\nPlease select the directory containing the set of pre-processed images.\n\n'
        'This should be the same directory you previously selected for the Morphology Analysis.\n\n'
    ), wraplength=750, justify="left", font=("Helvetica", 22))
    label.pack(pady=15)

    def select_directory():
        directory = filedialog.askdirectory()
        if directory:
            window.destroy()
            Spatial_vizualization_get_parameters(directory, regions, subjects, groups, treatments, tissues)

    button = tk.Button(window, text="Select Directory", command=select_directory, font=("Helvetica", 16))
    button.pack(pady=20)
    
    def on_closing():
        window.destroy()
        create_main_window()  # Reopen the main window when this window is closed

    window.protocol("WM_DELETE_WINDOW", on_closing)  # Handle the window close event

    window.update_idletasks()
    window.minsize(label.winfo_width() + 40, label.winfo_height() + 150)

    window.mainloop()

    window.update_idletasks()
    window.minsize(label.winfo_width() + 40, label.winfo_height() + 150)

    window.mainloop()

def Spatial_vizualization_get_parameters(directory, regions, subjects, groups, treatments, tissues):
    input_window = tk.Tk()
    input_window.title("Input Parameters")
    center_window(input_window, 400, 400)

    tk.Label(input_window, text="Number of neighbors:", font=("Helvetica", 14)).pack(pady=5)
    n_neighbors_entry = tk.Entry(input_window, font=("Helvetica", 14))
    n_neighbors_entry.pack(pady=5)

    tk.Label(input_window, text="Minimum distance:", font=("Helvetica", 14)).pack(pady=5)
    min_dist_entry = tk.Entry(input_window, font=("Helvetica", 14))
    min_dist_entry.pack(pady=5)

    tk.Label(input_window, text="Minimum cluster size:", font=("Helvetica", 14)).pack(pady=5)
    min_cluster_size_entry = tk.Entry(input_window, font=("Helvetica", 14))
    min_cluster_size_entry.pack(pady=5)

    tk.Label(input_window, text="Minimum samples:", font=("Helvetica", 14)).pack(pady=5)
    min_samples_entry = tk.Entry(input_window, font=("Helvetica", 14))
    min_samples_entry.pack(pady=5)
    

    def submit():
        try:
            n_neighbors = int(n_neighbors_entry.get())
            min_dist = float(min_dist_entry.get())
            min_cluster_size = int(min_cluster_size_entry.get())
            min_samples = int(min_samples_entry.get())
            input_window.destroy()
            Spatial_vizualization_start_spatial_visualization(directory, n_neighbors, min_dist, min_cluster_size, min_samples, regions, subjects, groups, treatments, tissues)
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid values for all parameters.")

    tk.Button(input_window, text="Submit", command=submit, font=("Helvetica", 14)).pack(pady=10)

    input_window.mainloop()

def Spatial_vizualization_start_spatial_visualization(directory, n_neighbors, min_dist, min_cluster_size, min_samples, regions, subjects, groups, treatments, tissues):
    Spatial_vizualization_processing_window(directory, n_neighbors, min_dist, min_cluster_size, min_samples, regions, subjects, groups, treatments, tissues)

def Spatial_vizualization_processing_window(directory, n_neighbors, min_dist, min_cluster_size, min_samples, regions, subjects, groups, treatments, tissues):
    window = tk.Toplevel()
    window.title("Processing")
    center_window(window, 600, 300)

    label = tk.Label(window, text="Processing...", font=("Helvetica", 22, "bold"))
    label.pack(pady=10)

    log_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=10, font=("Helvetica", 12))
    log_text.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

    def log_callback(message):
        log_text.insert(tk.END, message + '\n')
        log_text.see(tk.END)

    def on_completion(returncode):
        global processing
        processing = False
        window.after(3000, lambda: Spatial_vizualization_show_completion_window(window, returncode))

    def update_label():
        while processing:
            for suffix in ['', '.', '..', '...']:
                label.config(text=f"Processing{suffix}")
                time.sleep(0.5)
                if not processing:
                    break

    global processing
    processing = True
    thread = threading.Thread(target=Spatial_vizualization_run_processing, args=(
        directory, n_neighbors, min_dist, min_cluster_size, min_samples, regions, subjects, groups, treatments, tissues, 
        log_callback, on_completion
    ))
    thread.start()

    animation_thread = threading.Thread(target=update_label)
    animation_thread.start()

    window.mainloop()

def Spatial_vizualization_show_completion_window(previous_window, returncode):
    previous_window.destroy()
    window = tk.Toplevel()
    window.title("Process Completed")
    center_window(window, 600, 250)

    if returncode == 0:
        title = "Cell Mapping Completed Successfully"
        message = '\nYou can find your results in the'
        message_path= '"MorphoGlia > ID_clusters" folder'

    else:
        title = "Error"
        message = "An error occurred during the process."

    label = tk.Label(window, text=title, font=("Helvetica", 22, "bold"))
    label.pack(pady=20)

    text_widget = tk.Label(window, text=message, wraplength=550, justify="center", font=("Helvetica", 19))
    text_widget.pack(pady=5)
    
    text_widget = tk.Label(window, text=message_path, wraplength=550, justify="center", font=("Helvetica", 19, "bold"))
    text_widget.pack(pady=5)

    window.update_idletasks()
    window.minsize(text_widget.winfo_width() + 40, text_widget.winfo_height() + 150)



#####################################################################################
############################### SPATIAL VIZUALIZATION   #############################
#####################################################################################



#####################################################################################
####################################     MAIN    ####################################
#####################################################################################

    
    
#MAIN GUI        

def on_closing():
    """Handle the window close event."""
    root.quit()
    root.destroy()




def show_transition(callback):
    transition_window = tk.Toplevel()  # Use Toplevel instead of Tk
    transition_window.title("MorphoGlia")
    transition_window.attributes("-topmost", True)
    transition_window.attributes("-alpha", 0.0)  # Start fully transparent
    center_window(transition_window, 800, 600)  # Center the window on the screen

    # Handle the close button
    transition_window.protocol("WM_DELETE_WINDOW", on_closing)

    # Load and display the image
    img_path = resource_path('resources/transition_image.png')  # Adjust path accordingly
    image = PhotoImage(file=img_path)
    image_label = tk.Label(transition_window, image=image)
    image_label.pack(expand=True, pady=(50, 0))  # Padding above the image, no padding below

    label = tk.Label(transition_window, text="", font=("Helvetica", 48, "bold"))
    label.pack(expand=True, pady=(0, 50))  # Small padding above the text, and some below

    def close_and_callback():
        transition_window.destroy()
        callback()

    # Start the fade-in for the window and text
    fade_in_window(transition_window, close_and_callback, steps=30, delay=20)
    fade_in_text(label, "MorphoGlia", steps=100, delay=60)

    transition_window.mainloop()









def welcome_window():
    window = tk.Toplevel()  # Use Toplevel so it's not the root window
    window.title("MorphoGlia")
    center_window(window, 800, 600)

    # Handle the close button
    window.protocol("WM_DELETE_WINDOW", on_closing)

    label = tk.Label(window, text="Welcome to MorphoGlia", font=("Helvetica", 22, "bold"))
    label.pack(pady=20)
    
    text = (
        "MorphoGlia is an interactive tool designed to differentiate between study groups, identifying distinct microglial morphology clusters and mapping microglia cells onto tissue.\n\n"
        "It employs a machine learning ensemble to select relevant morphological features, perform dimensionality reduction, and cluster these features to color-code the cells.\n\n"
        "The color-coded cells are mapped back onto tissue microphotographs to visualize their spatial arrangement.\n\n"
        "This approach serves two main purposes: confirming similarities among cells within the same cluster and providing insights into the spatial distribution of each clustered cell.\n\n"
        "This visualization is particularly useful for spatial analyses, allowing for the identification of the most affected zones and uncovering previously unexplored patterns in disease physiopathology."
    )
        
    label = tk.Label(window, text=text, wraplength=750, justify="left", font=("Helvetica", 20))
    label.pack(pady=10)

    button = tk.Button(window, text="Continue", command=lambda: proceed_to_main_window(window), font=("Helvetica", 16))
    button.pack(pady=20)

    window.update_idletasks()
    window.minsize(label.winfo_width() + 40, label.winfo_height() + 150)

    window.mainloop()
    
    
    

def proceed_to_main_window(welcome_win):
    welcome_win.destroy()
    create_main_window()





def create_main_window():
    window = tk.Toplevel()  # Use Toplevel instead of Tk
    window.title("MorphoGlia")
    window.attributes("-alpha", 0.0)  # Start fully transparent
    center_window(window, 800, 600)  # Center the window on the screen

    # Handle the close button
    window.protocol("WM_DELETE_WINDOW", on_closing)

    # Set the window icon using a PNG image with iconphoto
    try:
        img = tk.PhotoImage(file=resource_path('resources/transition_image_title.png'))
        print(f"Image loaded: {img}")  # Debugging line
        window.iconphoto(True, img)
    except Exception as e:
        print(f"Failed to load icon: {e}")

    label = tk.Label(window, text="Select the Analysis", font=("Helvetica", 27, "bold"))
    label.pack(pady=30)
    
    morpho_button = tk.Button(window, text="Morphology Analysis", font=("Helvetica", 20), command=lambda: morphology_nomenclature_window(window))
    morpho_button.pack(pady=15)

    rfe_button = tk.Button(window, text="Feature Selection (RFE)", font=("Helvetica", 20), command=lambda: RFE_show_initial_message())
    rfe_button.pack(pady=15)

    umap_button = tk.Button(window, text="Dimensionality Reduction (UMAP)", font=("Helvetica", 20), command=UMAP_show_initial_message)
    umap_button.pack(pady=15)

    hdbscan_button = tk.Button(window, text="Clustering (HDBSCAN)", font=("Helvetica", 20), command=HDBSCAN_show_initial_message)
    hdbscan_button.pack(pady=15)

    spatial_viz_button = tk.Button(window, text="Spatial Visualization", font=("Helvetica", 20), command=lambda: Spatial_vizualization_nomenclature_window())
    spatial_viz_button.pack(pady=15)

    fade_in_main_window(window, steps=20, delay=15)






def fade_in_main_window(window, steps=20, delay=15):
    def fade(step):
        alpha = step / float(steps)
        window.attributes("-alpha", alpha)
        if step < steps:
            window.after(delay, fade, step + 1)

    fade(0)




def center_window(window, width=800, height=600):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = int((screen_width / 2) - (width / 2))
    y = int((screen_height / 2) - (height / 2))
    window.geometry(f"{width}x{height}+{x}+{y}")

def fade_in_window(window, callback, steps=30, delay=20):
    def fade(step):
        alpha = step / float(steps)
        window.attributes("-alpha", alpha)
        if step < steps:
            window.after(delay, fade, step + 1)
        else:
            window.after(3000, callback)

    fade(0)




def fade_in_text(label, text, steps=100, delay=60):
    def fade_step(step):
        gray_value = int(step * (255 / steps))
        color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'
        label.config(text=text, fg=color)
        if step < steps:
            label.after(delay, fade_step, step + 1)

    fade_step(0)
  
    
  
    
  
def create_app():
    global root
    root = tk.Tk()  # Create the root window only once
    root.withdraw()  # Hide the root window since we don't need it directly
    show_transition(welcome_window)




if __name__ == "__main__":
    create_app()




#####################################################################################
####################################     MAIN    ####################################
#####################################################################################


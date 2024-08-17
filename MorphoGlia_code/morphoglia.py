#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:02:46 2023

@author: juanpablomayaarteaga
"""

import os
import tifffile as tiff
import re
from skimage import io

def name_2_num(path=None):
    if path is None:
        raise ValueError("The 'path' argument is required.")
    
        
    for filename in os.listdir(path):
        if filename.endswith(".tif"):
            original_path = os.path.join(path, filename)
            just_number = re.findall(r"\d+", filename)[0] + ".tif"  
            new_path = os.path.join(path, just_number)
            os.rename(original_path, new_path) 
            
            
            


def name_to_number(path=None):

    for filename in os.listdir(path):
        if filename.endswith(".tif") and "_cell_" in filename:
            # Split the filename by "_cell_" and take the second part
            cell_number = filename.split("_cell_")[-1].split(".")[0]  # Extract the number
            
            # Construct the new filename with just the extracted number
            new_filename = f"{cell_number}.tif"
            
            # Rename the file
            os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
            print(f"Renamed: {filename} to {new_filename}")

            
    
        
        
        
        
        
def save_tif(filename, name=None, path=None, variable=None):
    if name is None:
        name = "add_name"  
    if path is None:
        path = "default/path"
    if variable is None:
        raise ValueError("The 'variable' argument is required.")

    output_filename = filename[:-4] + name
    output_path = os.path.join(path, output_filename)
    tiff.imwrite(output_path, variable)
    


def set_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        
    return path


import numpy as np
import cv2



def gammaCorrection(image, gamma):

    table = [((i / 255) ** gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(image, table)



from PIL import Image

def padding(image_path, padding_rows, padding_cols):
    # Load the image
    image = Image.open(image_path).convert("L")

    # Get the dimensions of the original image
    width, height = image.size

    # Create a new image with the desired dimensions
    new_width = width + 2 * padding_cols
    new_height = height + 2 * padding_rows
    new_image = Image.new("L", (new_width, new_height), 0)

    # Paste the original image into the new image, leaving the padding area empty
    new_image.paste(image, (padding_cols, padding_rows))

    # Save the modified image
    new_image.save("padded_image.tiff")
    padded_image=np.array(new_image)

    print("Padding added successfully.")
    return padded_image

def find_contours(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours





def polygon(image):
    # Find the contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate the contours to reduce points
    epsilon = 0.0001 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # Create a convex hull around the approximated contour
    hull = cv2.convexHull(approx)

    # Create a blank image
    polygon_image = np.zeros_like(image, dtype=np.uint8)

    # Draw the convex hull on the blank image
    cv2.fillPoly(polygon_image, [hull], (255))

    return polygon_image




def calculate_area(image):
    # Read the  image
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to convert to binary image
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

    # Count the white pixels (foreground)
    area = cv2.countNonZero(thresholded_image)

    return area







def cell_analysis(image_path):
    # Read the  image
    cell_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to convert to binary image
    _, thresholded_image = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY)

    # Count the white pixels (foreground)
    Cell_area = cv2.countNonZero(thresholded_image)
    
    if Cell_area != 0:
        contours = find_contours(image_path)  
        
        cell_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the eccentricity
        if len(cell_contour) >= 5:
            _, (major_axis, minor_axis), _ = cv2.fitEllipse(cell_contour)
            Cell_eccentricity = major_axis / minor_axis
        
        
        # Calculate the bounding box of the contour
        x, y, width, height = cv2.boundingRect(cell_contour)
        
        # Calculate the aspect ratio
        Cell_aspect_ratio = width / float(height)
        
        
        # Calculate the Feret diameter (maximum caliper)
        min_rect = cv2.minAreaRect(cell_contour)
        Cell_feret_diameter = max(min_rect[1])  # Maximum of width and height
        
        # Calculate the orientation (angle of the Feret diameter)
        Cell_orientation = min_rect[2]  # Angle in degrees
        
        # Calculate the perimeter
        Cell_perimeter = 0
        for contour in contours:
            Cell_perimeter += cv2.arcLength(contour, closed=True) 
        
        # Calculate circularity and compactness only if the perimeter is non-zero
        if Cell_perimeter != 0:
            Cell_circularity = 4 * np.pi * (Cell_area / (Cell_perimeter ** 2))
            Cell_compactness = Cell_area / Cell_perimeter
    
    
    else:
        Cell_area = 0
        Cell_feret_diameter = 0
        Cell_orientation = 0
        Cell_perimeter = 0
        Cell_circularity = 0 
        Cell_compactness = 0
        Cell_eccentricity = 0
        Cell_aspect_ratio = 0
    

    return Cell_area, Cell_perimeter, Cell_circularity, Cell_compactness, Cell_orientation, Cell_feret_diameter, Cell_eccentricity, Cell_aspect_ratio











def soma_analysis(image_path):
    # Read the image
    soma_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to convert to binary image
    _, thresholded_soma_image = cv2.threshold(soma_image, 0, 255, cv2.THRESH_BINARY)

    # Count the white pixels (foreground)
    Soma_area = cv2.countNonZero(thresholded_soma_image)
    
    # Initialize variables
    Soma_perimeter = 0
    Soma_circularity = 0
    Soma_compactness = 0
    Soma_orientation = 0
    Soma_feret_diameter = 0
    Soma_aspect_ratio = 0
    Soma_eccentricity = 0
    
    if Soma_area != 0:
        contours = find_contours(image_path)  
        
        soma_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the eccentricity
        if len(soma_contour) >= 5:
            _, (major_axis, minor_axis), _ = cv2.fitEllipse(soma_contour)
            Soma_eccentricity = major_axis / minor_axis
        
        
        # Calculate the bounding box of the contour
        x, y, width, height = cv2.boundingRect(soma_contour)
        
        # Calculate the aspect ratio
        Soma_aspect_ratio = width / float(height)
        
        
        # Calculate the Feret diameter (maximum caliper)
        min_rect = cv2.minAreaRect(soma_contour)
        Soma_feret_diameter = max(min_rect[1])  # Maximum of width and height
        
        # Calculate the orientation (angle of the Feret diameter)
        Soma_orientation = min_rect[2]  # Angle in degrees
        
        # Calculate the perimeter
        for contour in contours:
            Soma_perimeter += cv2.arcLength(contour, closed=True) 
        
        # Calculate circularity and compactness only if the perimeter is non-zero
        if Soma_perimeter != 0:
            Soma_circularity = 4 * np.pi * (Soma_area / (Soma_perimeter ** 2))
            Soma_compactness = Soma_area / Soma_perimeter
    
    return Soma_area, Soma_perimeter, Soma_circularity, Soma_compactness, Soma_orientation, Soma_feret_diameter, Soma_eccentricity, Soma_aspect_ratio










from scipy.ndimage import label, generate_binary_structure

def erase(image, elements):
    
    x=elements
    structure=generate_binary_structure(2,2)
    labeled_image, num_labels = label(image, structure)
    label_sizes=np.bincount(labeled_image.ravel())
    mask=label_sizes[labeled_image]>x
    erased_image=np.where(mask, image, 0)
    
    erased_image = (erased_image * 255).astype(np.uint8)
    
    return erased_image





def detect_and_color(image):
    # Padding
    img_padded = np.pad(image, 1)

    # Detect columns and rows
    n_row = np.shape(img_padded)[0]
    n_col = np.shape(img_padded)[1]

    # Empty matrix
    M = np.zeros((n_row - 2, n_col - 2), dtype=np.int8)

    # Create an empty colored image
    colored_image = np.zeros((M.shape[0], M.shape[1], 3), dtype=np.uint8)

    # Detect neighbors and assign colors
    for i in range(1, n_row):
        for j in range(1, n_col):
            if img_padded[i, j] != 0:
                M[i-1, j-1] = np.count_nonzero(img_padded[i-1:i+2, j-1:j+2]) - 1  # matriz 3x3

                value = M[i-1, j-1]
                if value == 1:
                    colored_image[i-1, j-1] = (0, 0, 255)  # Blue - Red (tif)
                elif value == 2:
                    colored_image[i-1, j-1] = (255, 0, 0)  # Red - Blue (tif)
                elif value >= 3:
                    colored_image[i-1, j-1] = (0, 255, 0)  # Green - Green (tif)
            

    # Converting to RGB
    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    return M, colored_image





def count_branches(image):
    image= image/255
    height = len(image)
    width = len(image[0])
    visited = [[False] * width for _ in range(height)]
    label = 0
    object_count = 0

    def dfs(row, col):
        visited[row][col] = True

        # Check neighbors (horizontally, vertically, and diagonally)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                new_row = row + dr
                new_col = col + dc

                # Skip out-of-bounds pixels
                if new_row < 0 or new_row >= height or new_col < 0 or new_col >= width:
                    continue

                # If connected foreground pixel is found, perform DFS on it
                if image[new_row][new_col] == 1 and not visited[new_row][new_col]:
                    dfs(new_row, new_col)

    for row in range(height):
        for col in range(width):
            if image[row][col] == 1 and not visited[row][col]:
                label += 1
                object_count += 1
                dfs(row, col)

    return object_count


from scipy.ndimage import label, generate_binary_structure

def count(image):
    
    labels, num_objects = label(image)
    
    return num_objects


def detect_features(image, soma_image):
    
    kernel = np.ones((3,3),np.uint8)
    soma_image = cv2.dilate(soma_image, kernel, iterations=1)
    # Padding
    img_padded = np.pad(image, 1)

    # Detect columns and rows
    n_row = np.shape(img_padded)[0]
    n_col = np.shape(img_padded)[1]

    # Empty matrix
    M = np.zeros((n_row - 2, n_col - 2), dtype=np.int8)

    # Create an empty colored image
    colored_image = np.zeros((M.shape[0], M.shape[1], 3), dtype=np.uint8)

    # Detect neighbors and assign colors
    for i in range(1, n_row):
        for j in range(1, n_col):
            if img_padded[i, j] != 0:
                M[i-1, j-1] = np.count_nonzero(img_padded[i-1:i+2, j-1:j+2]) - 1  # matriz 3x3

                value = M[i-1, j-1]
                if value == 1:
                    colored_image[i-1, j-1] = (0, 0, 255)  # Blue - Red (tif)
                elif value == 2:
                    colored_image[i-1, j-1] = (255, 0, 0)  # Red - Blue (tif)
                elif value >= 3:
                    colored_image[i-1, j-1] = (0, 255, 0)  # Green - Green (tif)

    # Converting to RGB
    colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)

    # Identify initial points in contact with soma
    initial_points = cv2.bitwise_and(image, soma_image)
    colored_image[initial_points > 0] = (200, 200, 0)  # Yellow (tif)

    return M, colored_image

                       
def fractal_dimension(image):
    
    # Threshold the image, convert non-black pixels to True and black pixels to False
    thresholded_img = image > 0

    # Sizes of boxes to use, scales as powers of 2
    sizes = [2**i for i in range(4, 10)]
    
    # List to hold counts of boxes at different sizes
    counts = []
    
    # For each box size
    for size in sizes:
        count = 0
        # Cover image with non-overlapping boxes
        for x in range(0, thresholded_img.shape[0], size):
            for y in range(0, thresholded_img.shape[1], size):
                # Check if there are any pixels in the box
                if np.any(thresholded_img[x:x+size, y:y+size]):
                    count += 1
        counts.append(count)
    
    # Calculate slopes of log-log plot to estimate fractal dimension
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return coeffs[0]




                          

import cv2
import numpy as np
from skimage.measure import label as labels, regionprops

def sholl_circles(image):
    # Extract the red channel
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds for the red color
    lower_red = np.array([0, 0, 200], dtype=np.uint8)  
    upper_red = np.array([100, 100, 255], dtype=np.uint8)  

    # Create a mask to filter out the white values
    mask_white = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    # Create a mask to filter out the red values
    mask_red = cv2.inRange(img, lower_red, upper_red)

    # Combine the masks
    red_pixels = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask_white))

    # Connected components labeling
    centroid, num_objects = labels(red_pixels, connectivity=2, background=0, return_num=True)

    # Get region properties
    props = regionprops(centroid)

    # Calculate the distance between the centroid and the four vertices of the image
    max_distance = max(
        np.linalg.norm([props[0].centroid[1], props[0].centroid[0]]),  # Top-left vertex
        np.linalg.norm([props[0].centroid[1], img.shape[0] - props[0].centroid[0]]),  # Bottom-left vertex
        np.linalg.norm([img.shape[1] - props[0].centroid[1], props[0].centroid[0]]),  # Top-right vertex
        np.linalg.norm([img.shape[1] - props[0].centroid[1], img.shape[0] - props[0].centroid[0]])  # Bottom-right vertex
    )

    # Pad the image to be exactly the size of the circumscribed circle
    padding = int(max_distance)  # Round down to the nearest integer
    img_padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)))

    # Create an empty colored image
    circle_image = np.zeros_like(img_padded)

    # Draw a blue point at the centroid
    blue_point_coords = (
        int(props[0].centroid[1] + padding), 
        int(props[0].centroid[0] + padding)
    )  # Row and column coordinates
    circle_image[blue_point_coords[1], blue_point_coords[0]] = [255, 0, 0]

    # Draw red growing circles with the centroid as the center
    for radius in range(25, int(max_distance), 10):
        cv2.circle(circle_image, blue_point_coords, radius, (0, 0, 255), 2)

    # Overlap the padded input image and the colored image bitwise
    result_image = cv2.bitwise_or(img_padded, circle_image)

    # Detect if the colored image and the padded input image touch at some points
    touching_points = cv2.bitwise_and(img_padded, circle_image)
    
    # Count the number of touching points
    num_touching_points = np.max(labels(touching_points))
    
    # Convert the touching points to green
    result_image[np.where((touching_points == [0, 0, 255]).all(axis=2))] = [0, 255, 0]  # Green color
    sholl_image = result_image
    #sholl_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    #circle_image = cv2.cvtColor(circle_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(sholl_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask
    _, thresholded = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the Sholl image to the bounding box
    sholl_image = sholl_image[y:y + h, x:x + w]
    # Crop the Sholl image to the bounding box
    circle_image = circle_image[y:y + h, x:x + w]
    
    return sholl_image, max_distance, num_touching_points, circle_image



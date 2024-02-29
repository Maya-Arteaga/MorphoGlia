# Load necessary libraries
library(rstatix)

# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/")

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_15_0.01_20_20.csv")

# List of variables to create distribution plots for
variables <- c('Soma_area', 'Soma_perimeter', 'Soma_circularity', 'Soma_compactness', 
               'Soma_feret_diameter', 'Soma_eccentricity', 'Soma_aspect_ratio', 
               'End_Points', 'Junctions', 'Branches', 'Initial_Points', 'Total_Branches_Length', 
               'ratio_branches', 'Convex_Hull_area', 'Convex_Hull_perimeter', 
               'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Convex_Hull_feret_diameter',
               'Cell_area', 'Cell_perimeter', 'Cell_circularity', 'Cell_compactness', 
               'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_max_distance', 
               'Sholl_crossing_processes', 'Sholl_circles', 'Cell_solidity', 'Cell_convexity', "UMAP_1", "UMAP_2")

# Perform Shapiro-Wilk test for normality for each variable
for (variable in variables) {
  test_result <- shapiro_test(data[, variable])
  print(paste("Variable:", variable))
  print(test_result)
}

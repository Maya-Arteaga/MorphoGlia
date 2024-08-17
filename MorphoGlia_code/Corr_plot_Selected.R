# Load necessary libraries
library(corrplot)

# Set the working directory
setwd("/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/")

# Define output directory for correlation plots
Plot_path <- file.path("Output_images", "Plots", "Corr")

# Ensure the directory exists
if (!file.exists(Plot_path)) {
  dir.create(Plot_path, recursive = TRUE)
}

# Set the filename for the plot
plot_filename <- file.path(Plot_path, "corr_plot_Selected_features.png")

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_10_0.1_20_10.csv")

# Define the desired DPI (e.g., 300)
dpi <- 1200

# Calculate the width and height of the image in pixels
width_pixels <- dpi * 10
height_pixels <- dpi * 10

# Define the features to assess
features_to_assess <- c('Soma_area', 'Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 
                         'End_Points', 'Branches', 'Convex_Hull_eccentricity', 
                         'Cell_area', 'Cell_compactness', 'Cell_feret_diameter', 'Cell_eccentricity', 
                         'Cell_aspect_ratio', 'Sholl_max_distance', 'Cell_solidity', 'Cell_convexity')



# Ensure all features are available in the dataset
if (!all(features_to_assess %in% names(data))) {
  stop("One or more specified features are not present in the dataset.")
}

# Calculate the correlation matrix, handling missing values
cor_matrix <- cor(data[features_to_assess], use = "complete.obs")

# Check if the correlation matrix is square (which it should be)
if (nrow(cor_matrix) != ncol(cor_matrix)) {
  stop("The correlation matrix is not square, which indicates a potential issue in the data.")
}

# Replace underscores with spaces in the feature names for better readability in the plot
feature_labels <- gsub("_", " ", features_to_assess)

# Apply modified labels to matrix dimensions for corrplot
rownames(cor_matrix) <- feature_labels
colnames(cor_matrix) <- feature_labels

# Open a PNG graphics device with the specified dimensions
png(filename = plot_filename, width = width_pixels, height = height_pixels, res = dpi)

# Use corrplot to visualize the correlation matrix
corrplot(
  cor_matrix,
  #addCoef.col = 'black',  # Color for the correlation coefficients
  type = "upper",  # Show only the upper part of the matrix
  method = "circle",  # Use circles to represent correlations
  #number.cex = 0.4,  # Size of the correlation coefficients
  font = 2,  # Font type for text
  tl.pos = 'd',  # Text labels position (down)
  tl.cex = 0.4,  # Size of text labels
  tl.col = "black",  # Color of text labels
  col = COL2('RdBu', 10)  # Color palette
)
dev.off()  # Close the graphics device
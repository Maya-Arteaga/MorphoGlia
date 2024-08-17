# Load necessary libraries
library(corrplot)


# Set the working directory
setwd("/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/")


# Define output directory for corr plots
Plot_path <- file.path("Output_images", "Plots", "Corr")

# Create directory if it doesn't exist
if (!file.exists(Plot_path)) {
  dir.create(Plot_path, recursive = TRUE)
}

# Set the filename for the plot
plot_filename <- file.path(Plot_path, "corr_plot_All.png")

# Load the data from the CSV file
#data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_15_0.01_20_20.csv")
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_10_0.1_20_10.csv")

# Define the desired DPI (e.g., 300)
dpi <- 1200
# Calculate the width and height of the image based on the desired DPI
width_inches <- 10  # Adjust as needed
height_inches <- 10  # Adjust as needed

# Calculate the width and height in pixels
width_pixels <- dpi * width_inches
height_pixels <- dpi * height_inches

# Open a PNG graphics device with the desired width, height, and DPI
png(plot_filename, width = width_pixels, height = height_pixels, res = dpi)


variables <- c('Soma_area', 'Soma_perimeter', 'Soma_circularity', 
                'Soma_compactness', 'Soma_orientation', 'Soma_feret_diameter', 
                'Soma_eccentricity', 'Soma_aspect_ratio', 'End_Points', 
                'Junctions', 'Branches', 'Initial_Points', 'Total_Branches_Length', 
                'ratio_branches', 'Convex_Hull_area', 'Convex_Hull_perimeter', 
                'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Fractal_dimension',
                'Convex_Hull_feret_diameter',
                'Cell_area', 'Cell_perimeter', 'Cell_circularity', 
                'Cell_compactness', 'Cell_feret_diameter', 
                'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_max_distance', 
                'Sholl_crossing_processes', 'Sholl_circles', 'Cell_solidity', 'Cell_convexity')



# Calculate the correlation matrix
cor_matrix <- cor(data[variables])

# Replace underscores with spaces in variable names for the correlation matrix
rownames(cor_matrix) <- gsub("_", " ", rownames(cor_matrix))
colnames(cor_matrix) <- gsub("_", " ", colnames(cor_matrix))

# Use corrplot to visualize the correlation matrix
corrplot(
  cor_matrix,
  #addCoef.col = 'black',  # Color for the correlation coefficients
  type = "lower",  # Show only the upper part of the matrix
  method = "circle",  # Use circles to represent correlations
  #number.cex = 0.4,  # Size of the correlation coefficients
  font = 2,  # Font type for text
  tl.pos = 'd',  # Text labels position (down)
  tl.cex = 0.2,  # Size of text labels
  tl.col = "black",  # Color of text labels
  col = COL2('RdBu'),  # Color palette
  cl.pos = 'n'  # Remove the legend
  #col = COL2('RdBu', 10)  # Color palette
)
dev.off()  # Close the graphics device



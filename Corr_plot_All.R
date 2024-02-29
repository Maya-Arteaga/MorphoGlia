# Load necessary libraries
library(corrplot)


# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/")

# Define output directory for corr plots
Plot_path <- file.path("Output_images", "Plots", "Corr")

# Create directory if it doesn't exist
if (!file.exists(Plot_path)) {
  dir.create(Plot_path, recursive = TRUE)
}

# Set the filename for the plot
plot_filename <- file.path(Plot_path, "corr_plot_All.png")

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_15_0.01_20_20.csv")

# Define the desired DPI (e.g., 300)
dpi <- 300
# Calculate the width and height of the image based on the desired DPI
width_inches <- 10  # Adjust as needed
height_inches <- 10  # Adjust as needed

# Calculate the width and height in pixels
width_pixels <- dpi * width_inches
height_pixels <- dpi * height_inches

# Open a PNG graphics device with the desired width, height, and DPI
png(plot_filename, width = width_pixels, height = height_pixels, res = dpi)


variables <- c('Soma_area', 'Soma_perimeter', 'Soma_circularity', 'Soma_compactness', 
               'Soma_feret_diameter', 'Soma_eccentricity', 'Soma_aspect_ratio', 
               'End_Points', 'Junctions', 'Branches', 'Initial_Points', 'Total_Branches_Length', 
               'ratio_branches', 'Convex_Hull_area', 'Convex_Hull_perimeter', 
               'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Convex_Hull_feret_diameter',
               'Cell_area', 'Cell_perimeter', 'Cell_circularity', 'Cell_compactness', 
               'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_max_distance', 
               'Sholl_crossing_processes', 'Sholl_circles', 'Cell_solidity', 'Cell_convexity')





# #@corrplot(cor(data[variables]))
# 
# corrplot.mixed(
#   cor(data[variables]),
#   upper="square",
#   lower="number",
#   addgrid.col="black",
#   tl.col="black",
#   tl.cex = 0.2, #Font size,
#   number.cex=0.6
# )
# 


corrplot(
  cor(data[variables]),
  addCoef.col = 'black',
  type = "upper",
  method = "circle",  # Specify the method parameter here
  number.cex=0.4,
  font = 2,
  tl.pos = 'd',
  tl.cex = 0.2,
  tl.col="black",
  col = COL2('RdBu', 10)
  
)

# Close the graphics device
dev.off()



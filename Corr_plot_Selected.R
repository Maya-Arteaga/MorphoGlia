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
plot_filename <- file.path(Plot_path, "corr_plot_Selected_square.png")

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




# Adjust the transparency of the default color palette
#adjusted_palette <- adjustcolor(default_palette, alpha.f = 1.2)

# Open a PNG graphics device with the desired width, height, and DPI
png(plot_filename, width = width_pixels, height = height_pixels, res = dpi)




features_to_assess <- c('Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 
                         'Junctions', 'Initial_Points', 'ratio_branches', 
                         'Convex_Hull_eccentricity', 
                         'Cell_compactness', 'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Cell_solidity', 'Cell_convexity',
                         'Sholl_circles')


# corrplot.mixed(
#   cor(data[features_to_assess]),
#   upper="number",
#   lower="circle",
#   addgrid.col="black",
#   tl.col="black",
#   tl.cex = 0.5, #Font size,
#   tl.srt=45,
#   number.cex=0.9,
#   font = 2,
#   number.font=2
# )

corrplot(
  cor(data[features_to_assess]),
  addCoef.col = 'black',
  type = "upper",
  method = "square",  # Specify the method parameter here
  number.cex=0.9,
  font = 2,
  tl.pos = 'd',
  tl.cex = 0.4,
  tl.col="black",
  col = COL2('RdBu', 10),
  tl.srt = 0
  
)


# Close the graphics device
dev.off()



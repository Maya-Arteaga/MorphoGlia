# Install and load necessary libraries
if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
  install.packages("ComplexHeatmap")
}
if (!requireNamespace("RColorBrewer", quietly = TRUE)) {
  install.packages("RColorBrewer")
}
if (!requireNamespace("dendextend", quietly = TRUE)) {
  install.packages("dendextend")
}
if (!requireNamespace("circlize", quietly = TRUE)) {
  install.packages("circlize")
}
library(ComplexHeatmap)
library(RColorBrewer)
library(dendextend)
library(circlize)

# Set the working directory
setwd("/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/")

# Define output directory for heatmap plots
Heatmap_path <- file.path("Output_images", "Plots", "Heatmap_R")
dir.create(Heatmap_path, recursive = TRUE, showWarnings = FALSE)

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_10_0.1_20_10.csv")

# Filter out data where Clusters is -1
data <- data[data$Clusters != -1, ]

# Define the variables to be used in the heatmap
variables <- c('Soma_area', 'Soma_perimeter', 'Soma_circularity', 'Soma_compactness', 
               'Soma_feret_diameter', 'Soma_eccentricity', 'Soma_aspect_ratio', 
               'End_Points', 'Junctions', 'Branches', 'Initial_Points', 'Total_Branches_Length', 
               'ratio_branches', 'Convex_Hull_area', 'Convex_Hull_perimeter', 
               'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Convex_Hull_feret_diameter',
               'Cell_area', 'Cell_perimeter', 'Cell_circularity', 'Cell_compactness', 'Fractal_dimension',
               'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_max_distance', 
               'Sholl_crossing_processes', 'Sholl_circles', 'Cell_solidity', 'Cell_convexity')

# Subset the data to include only the necessary variables and the Clusters column
data_filtered <- data[, c("Clusters", variables)]

# Replace underscores with spaces for attribute names
new_attribute_labels <- gsub("_", " ", variables)
colnames(data_filtered) <- c("Clusters", new_attribute_labels)

# Ensure Clusters column is a factor with specified levels
data_filtered$Clusters <- factor(data_filtered$Clusters, levels = c(0, 1, 2, 3, 4, 5))

# Scale the data for each column (variable)
scaled_data <- data_filtered
scaled_data[, -1] <- apply(data_filtered[, -1], 2, scale)

# Create a heatmap of the scaled features by Cluster
mean_data <- aggregate(. ~ Clusters, data = scaled_data, FUN = mean)

# Ensure the clusters are sorted according to hue order
mean_data <- mean_data[order(mean_data$Clusters), ]

# Transpose the data to have Clusters as columns and features as rows
heatmap_data <- t(as.matrix(mean_data[, -1]))
rownames(heatmap_data) <- new_attribute_labels
colnames(heatmap_data) <- paste("Cluster", mean_data$Clusters, sep=" ")

# Perform clustering on the transposed data
cluster_result <- hclust(dist(t(heatmap_data)))

# Convert the clustering result into a dendrogram
dend <- as.dendrogram(cluster_result)

# Manually reorder the columns to the desired order (0-5)
ordered_clusters <- colnames(heatmap_data)[order(match(colnames(heatmap_data), paste("Cluster", c(0, 1, 2, 3, 4, 5), sep=" ")))]
heatmap_data <- heatmap_data[, ordered_clusters]

# Reorder the dendrogram to match the manual column order
dend <- rotate(dend, ordered_clusters)

# Define the color scale from -1 to 1 using the RdBu palette
col_fun <- colorRamp2(seq(-1, 1, length.out = 50), colorRampPalette(RColorBrewer::brewer.pal(10, "RdBu"))(50))

# Define the heatmap object with larger font sizes and adequate spacing
ht <- Heatmap(heatmap_data, 
              cluster_rows = FALSE, 
              cluster_columns = dend, # Use the ordered dendrogram
              show_column_names = TRUE, 
              show_row_names = TRUE, 
              row_names_side = "right", # Display row (variable) names on the right
              row_names_gp = gpar(fontsize = 17, fontface = "bold"), # Increase row names size
              column_names_gp = gpar(fontsize = 20, fontface = "bold", just = "center"), # Center column names
              column_names_rot = 0, # Rotate column names to horizontal
              show_heatmap_legend = FALSE, # Remove the legend
              row_names_max_width = unit(10, "cm"), # Increase space for row names
              column_names_centered = TRUE, # Ensure column names are centered
              col = col_fun,
              column_title = "Heatmap of Scaled Features by Clusters",
              column_title_gp = gpar(fontsize = 24, fontface = "bold", just = "center"), # Increase column title size
              cell_fun = function(j, i, x, y, width, height, fill) {
                grid.rect(x = x, y = y, width = width, height = height, gp = gpar(col = NA, fill = fill))
              })

# Draw the heatmap with very high resolution (1200 dpi)
png(filename = file.path(Heatmap_path, "heatmap_with_dendrogram_Clusters_Scale.png"), 
    width = 16000, height = 20000, res = 1200) # Adjust width and height to maintain aspect ratio
draw(ht) # Draw the heatmap without the legend
dev.off()

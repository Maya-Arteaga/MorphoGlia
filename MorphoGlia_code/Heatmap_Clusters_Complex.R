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
setwd("/Users/juanpablomayaarteaga/Desktop/Hippocampus/Merge/Prepro/")

# Define output directory for heatmap plots
Heatmap_path <- file.path("Output_images", "Plots", "Heatmap_R")
dir.create(Heatmap_path, recursive = TRUE, showWarnings = FALSE)

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_10_0.01_20_20.csv")

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
col_fun <- colorRamp2(c(-1, 0, 1), colorRampPalette(brewer.pal(10, "RdBu"))(3))

# Define the heatmap object
ht <- Heatmap(heatmap_data, 
              cluster_rows = FALSE, 
              cluster_columns = dend, # Use the ordered dendrogram
              show_column_names = TRUE, 
              show_row_names = TRUE, 
              column_names_gp = gpar(fontsize = 22, fontface = "bold", just="center"), # Make column names bold
              column_names_rot = 0, # Rotate column names to horizontal
              row_names_gp = gpar(fontsize = 17),
              col = col_fun,
              column_title = "Heatmap of Scaled Features by Cluster",
              column_title_gp = gpar(fontsize = 20, fontface = "bold", just="center"), # Make column title bold
              heatmap_legend_param = list(title = "Value", title_gp = gpar(fontsize = 12, fontface = "bold"),
                                          labels_gp = gpar(fontsize = 10)),
              cell_fun = function(j, i, x, y, width, height, fill) {
                grid.rect(x = x, y = y, width = width, height = height, gp = gpar(col = NA, fill = fill))
              })

# Draw the heatmap
png(filename = file.path(Heatmap_path, "heatmap_with_dendrogram_Clusters.png"), width = 1200, height = 1400)
draw(ht, heatmap_legend_side = "left") # Position the legend on the left
dev.off()

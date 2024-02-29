#https://www.datanovia.com/en/lessons/heatmap-in-r-static-and-interactive-visualization/
#https://r-charts.com/correlation/pheatmap/
#https://www.rdocumentation.org/packages/pheatmap/versions/1.0.12/topics/pheatmap


#install.packages("pheatmap")
library("pheatmap")

# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/")

# Define output directory for corr plots
Plot_path <- file.path("Output_images", "Plots", "Hierarchical_Distances")

# Create directory if it doesn't exist
if (!file.exists(Plot_path)) {
  dir.create(Plot_path, recursive = TRUE)
}

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_15_0.01_20_20.csv")


#set variables to asses
variables <- c('Soma_area', 'Soma_perimeter', 'Soma_circularity', 'Soma_compactness', 
               'Soma_feret_diameter', 'Soma_eccentricity', 'Soma_aspect_ratio', 
               'End_Points', 'Junctions', 'Branches', 'Initial_Points', 'Total_Branches_Length', 
               'ratio_branches', 'Convex_Hull_area', 'Convex_Hull_perimeter', 
               'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Convex_Hull_feret_diameter',
               'Cell_area', 'Cell_perimeter', 'Cell_circularity', 'Cell_compactness', 
               'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_max_distance', 
               'Sholl_crossing_processes', 'Sholl_circles', 'Cell_solidity', 'Cell_convexity')



df <- scale(data[variables])

# Filter out rows with Cluster_Labels equal to -1
data <- data %>% filter(Cluster_Labels != -1)

# Convert Cluster_Labels column to factor if necessary
data$Cluster_Labels <- as.factor(data$Cluster_Labels)



# Set names for the clusters
names(data$Cluster_Labels) <- paste("Cluster", levels(data$Cluster_Labels))


# Calculate pairwise distances between clusters
cluster_distances <- dist(data$Cluster_Labels)

# Convert the distance object to a square matrix
cluster_dist_matrix <- as.matrix(cluster_distances)


# Plot the heatmap with unique data and annotation
pheatmap(cluster_dist_matrix,
         cluster_rows = TRUE,          
         cluster_cols = TRUE,          
         cutree_rows = 7,         
         cutree_cols = 7,              
         fontsize_row = 10,            
         fontsize_col = 10,            
         show_rownames = FALSE,        
         show_colnames = FALSE,  
         legend=TRUE,
         main = "Clusters Distance Heatmap",  
         filename = file.path(Plot_path, "clusters_distance_heatmap.png")
)




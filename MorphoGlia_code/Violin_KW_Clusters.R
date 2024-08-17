# Load necessary libraries
library(ggstatsplot)
library(PMCMRplus)
library(ggplot2)

# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hippocampus/Merge/Prepro/")

# Define output directory for violin plots
Violin_path <- file.path("Output_images", "Plots", "Clusters")

# Load the data from the CSV file
#data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_15_0.01_20_20.csv")
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_10_0.01_20_20.csv")

data <- data[data$Clusters != -1, ]

# Define category column
category_column <- "Clusters"

# Define hue order
hue_order <- c("0", "1", "2",  "3", "4", "5")

# Define colors
#colors <- c("#FF0000", "#FF8C00", "#FFFF00", "#008000", "#48D1CC", "#AFEEEE")
colors <- c("#500050", "#FF0000", "#FF8C00", "#FFFF00", "#008000", "#48D1CC")
#colors <- c("#500050", "#FF0000", "#FF8C00", "#FFFF00", "#008000", "#48D1CC", "#c2f0ee")

# List of variables to create distribution plots for
variables <- c('Soma_area', 'Soma_perimeter', 'Soma_circularity', 'Soma_compactness', 
               'Soma_feret_diameter', 'Soma_eccentricity', 'Soma_aspect_ratio', 
               'End_Points', 'Junctions', 'Branches', 'Initial_Points', 'Total_Branches_Length', 
               'ratio_branches', 'Convex_Hull_area', 'Convex_Hull_perimeter', 
               'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Convex_Hull_feret_diameter',
               'Cell_area', 'Cell_perimeter', 'Cell_circularity', 'Cell_compactness', 'Fractal_dimension',
               'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_max_distance', 
               'Sholl_crossing_processes', 'Sholl_circles', 'Cell_solidity', 'Cell_convexity', "UMAP_1", "UMAP_2")

# Convert category_column to factor
data[[category_column]] <- factor(data[[category_column]])

# Loop through each variable and generate a plot
for (variable in variables) {
  variable_code <- gsub("_", " ", variable)  # Replace underscores with spaces
  p <- ggbetweenstats(
    data = data,
    x = !!rlang::sym(category_column),
    y = !!rlang::sym(variable),
    type = "nonparametric",
    plot.type = "violin",
    pairwise.comparisons = TRUE,
    pairwise.display = "significant",
    p.adjust.method = "bonferroni",
    centrality.plotting = FALSE,
    bf.message = FALSE
  ) +
    ggtitle(paste("Kruskal-Wallis Test for", variable_code)) +
    labs(y = variable_code) +
    scale_color_manual(values = colors) +
    scale_x_discrete(limits = hue_order) +
    theme(
      text = element_text(size=15, family = "times new roman"),
      axis.text.x = element_text(size=15),
      plot.title = element_text(hjust = 0.5)
    )
  
  # Save the plot
  ggsave(filename = paste0(Violin_path, "/", variable, "_violin_plot.png"), plot = p, width = 12, height = 8, dpi = 300)
}

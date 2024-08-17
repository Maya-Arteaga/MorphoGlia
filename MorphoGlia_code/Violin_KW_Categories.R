# Load necessary libraries
library(ggstatsplot)
library(PMCMRplus)
library(ggplot2)

# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hippocampus/Merge/Prepro/")

# Define output directory for violin plots
Violin_path <- file.path("Output_images", "Plots", "Categories")

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_10_0.01_20_20.csv")

# Define category column
category_column <- "categories"

# Convert category_column to factor and replace underscores in category names
data[[category_column]] <- factor(gsub("_", " ", data[[category_column]]))

# Define a mapping of old category names to new names
new_category_names <- c("VEH SS CA1" = "Ctrl CA1", "VEH SCO CA1" = "Sco CA1",
                        "VEH SS HILUS" = "Ctrl Hilus", "VEH SCO HILUS" = "Sco Hilus")

# Apply the new category names mapping
levels(data[[category_column]]) <- new_category_names[levels(data[[category_column]])]

# Define hue order with new category names
hue_order <- c("Ctrl CA1", "Sco CA1", "Ctrl Hilus", "Sco Hilus")

# Define colors
colors <- c("#008000", "#FF0000", "#008080", "#FF8C00")

# List of variables to create distribution plots for
variables <- c('Soma_area', 'Soma_perimeter', 'Soma_circularity', 'Soma_compactness', 
               'Soma_feret_diameter', 'Soma_eccentricity', 'Soma_aspect_ratio', 
               'End_Points', 'Junctions', 'Branches', 'Initial_Points', 'Total_Branches_Length', 
               'ratio_branches', 'Convex_Hull_area', 'Convex_Hull_perimeter', 
               'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Convex_Hull_feret_diameter',
               'Cell_area', 'Cell_perimeter', 'Cell_circularity', 'Cell_compactness', 'Fractal_dimension',
               'Cell_feret_diameter', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_max_distance', 
               'Sholl_crossing_processes', 'Sholl_circles', 'Cell_solidity', 'Cell_convexity', "UMAP_1", "UMAP_2")

# Generate plots for each variable
for (variable in variables) {
  variable_code <- gsub(" ", "_", variable)  # Convert space to underscore for accessing data frame columns
  p <- ggbetweenstats(
    data = data,
    x = !!rlang::sym(category_column),
    y = !!rlang::sym(variable_code),
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
    scale_color_manual(values = setNames(colors, hue_order)) +  # Assign colors to clusters based on new category names
    scale_x_discrete(limits = hue_order) +  # Set order based on new category names
    theme(
      text = element_text(size=15, family = "Times New Roman"),
      axis.text.x = element_text(size=15),
      plot.title = element_text(hjust = 0.5)  # Center the title
    )
  
  # Save the plot
  ggsave(filename = paste0(Violin_path, "/", variable, "_violin_plot.png"), plot = p, width = 12, height = 8, dpi = 300)
}


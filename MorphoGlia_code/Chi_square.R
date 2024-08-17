# List of necessary libraries
libraries <- c("gplots", "ggplot2", "dplyr", "tidyr")

# Install any libraries that are not already installed
for (lib in libraries) {
  if (!requireNamespace(lib, quietly = TRUE)) {
    install.packages(lib)
  }
}

# Install corrplot package if not already installed
if (!requireNamespace("corrplot", quietly = TRUE)) {
  install.packages("corrplot")
}

# Load corrplot library
library(corrplot)

# Load the libraries
lapply(libraries, library, character.only = TRUE)

# Set the working directory and define paths
setwd("/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/")
Plot_path <- file.path("Output_images", "Plots", "Chi_R")
if (!file.exists(Plot_path)) dir.create(Plot_path, recursive = TRUE)

# Load data and prepare it
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_10_0.1_20_10.csv")
data <- data[data$Clusters != -1, ]  # Filter out noise
data$Cluster_Labels <- factor(data$Clusters)
data$categories <- factor(gsub("_", " ", data$categories))  # Replace underscores

# Define new category names and apply them
new_category_names <- c("VEH SS CA1" = "SS CA1", "VEH SCO CA1" = "SCOP CA1",
                        "VEH SS HILUS" = "SS Hilus", "VEH SCO HILUS" = "SCOP Hilus")
data$categories <- factor(data$categories, levels = names(new_category_names), labels = new_category_names)

# Create and print the contingency table
contingency_table <- table(data$categories, data$Clusters)
print(contingency_table)

# Chi-square test
chisq_results <- chisq.test(contingency_table)
print(chisq_results)

# Visualization: Balloon Plot
png(filename = file.path(Plot_path, "Cluster_Frequencies.png"), width = 800, height = 600)
balloonplot(t(contingency_table), main = "Frequencies", xlab = "Clusters", ylab = "Experimental Groups",
            label = TRUE, show.margins = FALSE, dotcolor = "lightblue", text.size = 1.5, font = 2)
dev.off()

# Visualization: Correlogram of Standard Residuals
png(filename = file.path(Plot_path, "Chi_Square_Correlogram.png"), width = 800, height = 600)
corrplot(chisq_results$residuals, is.cor = FALSE, tl.col = "black", tl.cex = 1.4, font = 2,
         addCoef.col = "black", cl.pos = "n", number.cex = 1.5, col = COL2('RdBu', 10), tl.srt = 0)
dev.off()

# Contribution Plot
contrib <- 100 * contingency_table^2 / sum(contingency_table)
contrib <- round(contrib, 3)
png(filename = file.path(Plot_path, "Contribution_Plot.png"), width = 800, height = 600)
corrplot(contrib, is.cor = FALSE, tl.col = "black", tl.cex = 1.2, font = 2,
         addCoef.col = "black", col = colorRampPalette(c("yellow", "red"))(10), cl.pos = "n", cl.cex = 1.8, tl.srt = 0)
dev.off()

#http://www.sthda.com/english/wiki/chi-square-test-of-independence-in-r
#http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram
#https://www.rdocumentation.org/packages/gplots/versions/3.1.3.1/topics/balloonplot

# Load necessary libraries
library(gplots)
# Load the corrplot package
library(corrplot)


# Set the working directory
setwd("/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/")

# Define output directory for corr plots
Plot_path <- file.path("Output_images", "Plots", "Chi_R")

# Create directory if it doesn't exist
if (!file.exists(Plot_path)) {
  dir.create(Plot_path, recursive = TRUE)
}

# Set the filename for the plot
plot_filename <- file.path(Plot_path, "Cluster_Frequencies.png")

# Load the data from the CSV file
data <- read.csv("Output_images/Merged_Data/Morphology_HDBSCAN_15_0.01_20_20.csv")

# Filter the data to include only the "categories" and "Cluster_Labels" columns
filtered_data <- data[, c("categories", "Cluster_Labels")]

# Filter out rows where "Cluster_Labels" equals -1
filtered_data <- filtered_data[filtered_data$Cluster_Labels != -1, ]


# Convert Cluster_Labels to factor
filtered_data$Cluster_Labels <- factor(filtered_data$Cluster_Labels)

# Check the structure of filtered_data
str(filtered_data)



############### OBSERVED FREQUENCIES

# Specify the desired order of categories
desired_order <- c("VEH_SS", "VEH_SCO", "CNEURO-01_SCO", "CNEURO-10_SCO", "CNEURO-10_SS")
# Replace underscores with spaces
#desired_order <- gsub("_", " ", desired_order)

# Create a contingency table
contingency_table <- table(filtered_data$categories, filtered_data$Cluster_Labels)

# Print the contingency table to verify the order
print(contingency_table)

"""
# Reorder the rows of the contingency table and convert the rows to factors with the desired order
contingency_table <- contingency_table[desired_order, ]
contingency_table <- as.data.frame.matrix(contingency_table)
contingency_table$category <- rownames(contingency_table)
rownames(contingency_table) <- NULL
contingency_table$category <- factor(contingency_table$category, levels = desired_order)
"""
# Reorder the rows of the filtered_data dataframe
filtered_data$categories <- factor(filtered_data$categories, levels = desired_order)



# Save the plot as a PNG file with specified dimensions
png(filename = plot_filename, width = 800, height = 600)  # Adjust width and height as needed
balloonplot(t(contingency_table),  # Transpose the contingency table
            main = "Frequencies", 
            ylab = "categories", xlab = "Cluster_Labels",  
            label = TRUE, 
            show.margins = FALSE,
            dotcolor = "lightblue",
            text.size = 1.5,
            font = 2
            )
dev.off()



############### CHI SQUARE - STANDARD RESIDUALS


# Perform chi-square test
chisq <- chisq.test(contingency_table)

# Extract observed frequencies
observed <- chisq$observed

# Extract and round residuals
residuals <- round(chisq$residuals, 3)

# Store the chi-square test result and residuals in a list
results <- list(
  chi_square_test = chisq,
  observed_frequencies = observed,
  residuals = residuals
)


# Set the filename for the correlogram plot
corplot_filename <- file.path(Plot_path, "chi_square_corplot.png")

# Save the correlogram plot as a PNG file
png(filename = corplot_filename, width = 800, height = 600)  # Adjust width and height as needed
corrplot(residuals, 
         is.cor = FALSE, 
         tl.col = "black", 
         tl.cex = 1.4, 
         font = 2, 
         addCoef.col = "black", 
         cl.pos = "n", 
         number.cex=1.5,
         col = COL2('RdBu', 10),
         tl.srt = 0
                    )
dev.off()




#######CONTRIBUTION PLOT

# Contribution in percentage (%)
contrib <- 100 * chisq$residuals^2 / chisq$statistic
contrib <- round(contrib, 3)

# Visualize the contribution
library(corrplot)
corplot_filename <- file.path(Plot_path, "contribution_corplot.png")

# Save the correlogram plot as a PNG file
png(filename = corplot_filename, width = 800, height = 600)  # Adjust width and height as needed
corrplot(contrib, is.cor = FALSE, tl.col = "black", tl.cex = 1.2, font = 2, addCoef.col = "black", cl.pos = "n", cl.cex = 1.8)
dev.off()






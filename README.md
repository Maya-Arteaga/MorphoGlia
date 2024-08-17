# MorphoGlia: Clustering and Mapping Microglia Morphology
                         
![GA5](https://github.com/user-attachments/assets/564cf540-9c39-4aa0-a68f-18a5e9076ea5)


## MorphoGlia code, Interface and Software
MorphoGlia has been developed with a focus on user-friendliness and accessibility, making it an ideal tool for the broader scientific community. The software is available in two main modes: a software mode and an interface mode, both designed to facilitate ease of use. The executable file was generated using PyInstaller (https://pyinstaller.org/en/stable/), while the interactive interface was built using the Tkinter Python library (https://docs.python.org/3/library/tkinter.html).
For advanced users, direct modification of the source code is recommended to tailor the application to the specific needs of individual experiments. This approach allows for greater flexibility and customization, ensuring that MorphoGlia can be adapted to a wide range of research scenarios.

To use the interface mode, download the MorphoGlia_Interface directory. You can run it from the terminal or through a Python interface. Ensure that the files MorphoGlia_app.py and morphoglia.py are located in the same directory. You will need to install the necessary libraries.

The executable mode is available for download at the following link: https://drive.google.com/drive/u/1/folders/15Mu2THZvVH6OTlDZuzf7ftsWyLAiWjUV

The source code is located in the MorphoGlia_code directory. This directory includes the interactive mode for point tracking, color mapping to customize cluster colors, and the R and Python scripts needed to reproduce the graphics.



![Figure2](https://github.com/Maya-Arteaga/Morphology/assets/70504322/c498a759-7cff-4317-ba91-7fa1a8c1521f)


### Morphology analysis
Each cell was identified in the complete binary photomicrographs, and classic morphometric features were computed using Python, primarily with the OpenCV library (https://pypi.org/project/opencv-python/). For skeleton analysis, the total branch length (in pixels), number of initial points (cell processes emerging from the soma), number of junction points (branch subdivisions), and number of endpoints (ends of branches) were measured. Cell body analysis included calculating the area, perimeter, circularity (with 1 representing a perfect circle), Feret diameter (maximum caliper diameter), compactness (how closely an object packs its area), aspect ratio (width/height), orientation (angle in degrees), and eccentricity (major axis/minor axis). The same metrics used in cell body analysis were applied to the entire cell. Fractal analysis involved determining convex hulls (the smallest convex set of pixels enclosing a cell) and performing the same calculations as in cell body analysis, as well as calculating the fractal dimension. Sholl analysis consisted of identifying the number of Sholl circles (circles with increasing radii created around the centroid of the cell soma), counting crossing processes (intersections of cell processes with Sholl circles), and measuring the maximum distance (distance between the centroid and the four vertices of the image). These metrics allow researchers to analyze the biologically relevant characteristics of the cell.

![Cell_Analysis](https://github.com/user-attachments/assets/8ef454ee-eb4b-41ab-993e-d8ff393ca723)


### Feature selection
Selecting the most appropriate features to characterize microglia is challenging due to significant biological variability depending on the region and pathology. A fixed set of features that best differentiates morphological states cannot be universally applied. To address this, a dynamic feature selection approach is necessary to ensure relevance and mitigate noise. We employed the Recursive Feature Elimination (RFE) algorithm, a specialized technique for selecting crucial features by iteratively reducing the feature set and removing the least important ones. RFE uses a Random Forest engine as the underlying training model to determine feature importance. The Random Forest algorithm ensures robustness, avoids overfitting, and captures non-linear relationships between features and the target variable. In this study, the groups (SS-CA1, SCOP-CA1, SS-Hilus, and SCOP-Hilus) were used as the target variable. The features analyzed were those computed in the morphology analysis, totaling 32 variables. The RFE algorithm selected the most significant half of these features, enhancing the robustness of the model and enabling feature selection suited to the specificities of each study group. This approach was implemented using the scikit-learn package (https://pypi.org/project/scikit-learn/).


![Fig2_corr_plot](https://github.com/user-attachments/assets/4fbe3d7b-7242-432a-a803-890d5bd38b93)


### Dimensionality reduction
Uniform Manifold Approximation and Projection (UMAP) is a technique designed for non-linear and non-parametric dimensionality reduction. This technique preserves both local and global structures of the data. It assumes that the data is uniformly distributed on a Riemannian manifold with a locally constant Riemannian metric and local connectivity [32, 33]. Key UMAP parameters include the number of nearest neighbors (n_neighbors) and the minimum distance between points (min_dist). The n_neighbors parameter constructs the high-dimensional neighborhood graph, with lower values focusing on local structures and higher values capturing global structures. Recommended values range from 5 to 50. The min_dist parameter controls point clustering, with lower values resulting in tighter clustering and higher values in more dispersed points. The recommended min_dist value is 0.1. Additionally, the number of components (n_components) determines the dimensionality for data reduction. UMAP uses fuzzy set theory to represent the probability distribution in both high-dimensional and low-dimensional spaces, preserving complex data patterns through non-linear embedding [32, 33]. This makes UMAP effective for capturing intricate relationships in the data. This approach was implemented using the scikit-learn package (https://pypi.org/project/umap-learn/). For further details consult https://github.com/lmcinnes/umap.

It is essential to adjust UMAP hyperparameters (n_neighbors and min_dist) based on the data characteristics and experimental goals. Various hyperparameters were tested, demonstrating robust results.

![UMAP_10_0 1](https://github.com/user-attachments/assets/6754a7af-70a6-4f2d-b0af-7e7e49e31a36)



### Clustering
Following UMAP for dimensionality reduction, Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) was used for clustering. This non-parametric method constructs a cluster hierarchy based on the multivariate modes of the underlying distribution by transforming the space according to density, building the cluster hierarchy, and extracting the clusters. HDBSCAN's density-based approach makes minimal assumptions about the clusters, identifying them as regions of high density separated by low-density regions, thus eliminating the need to specify the number of clusters beforehand. This method can identify clusters of varying shapes and sizes and creates a hierarchy based on different density levels. Key hyperparameters include the minimum cluster size (min_cluster_size) and minimum cluster samples (min_samples). The minimum cluster size determines the smallest number of points in a cluster for it to be considered valid; fewer points are treated as noise. Min_samples defines the minimum number of neighboring points required for a point to be considered a core point, which must include at least the specified number of sample points (including itself) in its neighborhood. HDBSCAN effectively handles data noise by excluding points that do not fall within high-density regions, making it robust to noise and outliers. This approach was implemented using the scikit-learn package (https://pypi.org/project/hdbscan/). For further details consult https://github.com/scikit-learn-contrib/hdbscan


![UMAP_HDBSCAN_10_0 1_20_10](https://github.com/user-attachments/assets/95d89ac5-d7c6-4cea-8f28-d5c1df14f32a)



### Confusion Matrices of the Study Groups and MorphoGlia Clusters
Contrasting the confusion matrices generated for distinguishing the study groups and MorphoGlia clusters reveals significant differences in classification accuracy. When evaluating the study groups directly, the accuracy is 0.47, indicating considerable overlap in morphological states. In contrast, clustering with the MorphoGlia pipeline significantly improves classification accuracy to 0.97. This improvement suggests that there is a mixing of morphological states within the study groups, akin to Simpson's Paradox, and that the MorphoGlia pipeline effectively unveils distinct morphological clusters.

![Confusion_Matrices](https://github.com/user-attachments/assets/63f70de7-a993-4649-95e3-bb74ff5cef8e)


### From Data Points to Microglial Morphologies: Visualizing Cluster Variability
This figure demonstrates the application of the umap.plot.interactive() function to trace individual microglial cells to their respective data points within the UMAP space. Despite the effective clustering of morphological states by HDBSCAN, there is variability within the clusters. This variability arises from the 16 selected features, which have been reduced to two dimensions through UMAP, capturing a broad spectrum of microglial morphologies. Notably, MorphoGlia displays these morphologies along a curved continuum, illustrating a gradual transition across diverse microglial states. This visualization underscores the complexity of microglial morphology and the utility of advanced clustering techniques in identifying subtle differences within grouped data.

![umap_cells_ink](https://github.com/user-attachments/assets/87c29efb-f4ab-45c8-a51f-2b68db83d1e4)



### Noise detection by HDBSCAN algorithm.
The Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm operates with minimal assumptions about cluster formation. It identifies clusters as regions of high density distinctly separated by low-density areas, eliminating the need to predefine the number of clusters. This functionality allows HDBSCAN to effectively manage data noise by excluding points outside high-density regions, thereby enhancing its robustness. This capability facilitates the detection of outliers or preprocessing errors. However, in this case, there seems to be no preprocessing error in the cell data. Accordingly, it may be inferred that the detected noise represents transitional morphological states between two clusters, which do not conform to any high-density regions. Further investigation is warranted to clarify this issue. The orange marker indicates the location of three data points represented as noise by HDBSCAN within the structure identified by UMAP. On the left, the corresponding three cells are displayed for visualization.


![Noise](https://github.com/user-attachments/assets/a8dbcf9b-1c22-4f7d-a6ba-1e0bd3e8f48e)



### Spatial analysis and visualization
Dimensionality reduction and clustering result in color-coded data points, each representing a cell. These color-coded cells are mapped back onto the tissue microphotograph to visualize their spatial arrangement. This approach serves two main purposes: confirming similarities among cells within the same cluster and providing insights into the spatial distribution of each clustered cell. This visualization is particularly useful for spatial analyses, allowing for the identification of the most affected zones and uncovering previously unexplored patterns in disease physiopathology.


![5](https://github.com/user-attachments/assets/7d7efaca-398b-478d-86c9-a44af693941d)






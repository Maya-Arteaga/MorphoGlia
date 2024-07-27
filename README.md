# MorphoGlia: A Microglia Morphology Classifier
                         
![Figure2](https://github.com/Maya-Arteaga/Morphology/assets/70504322/c498a759-7cff-4317-ba91-7fa1a8c1521f)

Introduction 

Since its discovery, microglia cells has intrigued researchers with their ability to respond to external stimuli, adapting its morphology, migrating, and accumulating at lesion sites (Del Rio-Hortega, 1932). This dynamic behavior endows them with remarkable sensitivity to various brain microenvironments, enabling them to execute a myriad of complex homeostatic processes and resulting in a remarkably diverse morphology (Matejuk, 2020; Sierra et al., 2014; Lier, J. 2020; McNamara et al., 2014 al., 2023). Due to their responsiveness to microenvironments, microglia play a crucial role in understanding and addressing neurological diseases and central nervous system disorders (Li, 2018; Matejuk, 2020). Consequently, the quantitative assessment of microglial morphology has become widespread in neuroscience, particularly in the evaluation of neuroinflammatory states (Green, 2022).

Given their sensitivity to microenvironmental cues, microglia display a wide range of morphologies, emphasizing the importance of detecting subtle changes as early indicators of their immediate response to local environmental signals (Stratoulias, 2019; Siegert, 2022; Reddaway, 2023). Hence, ensuring reliable inferences requires precise quantification of multiple microglial features to capture the variability in morphology for a biologically meaningful interpretation (Siegert, 2022; Reddaway, 2023). However, many analyses have been constrained by the limited number or types of features considered. The most widely used methods for analyzing microglial morphology include: 1) Skeleton analysis; 2) Cell body area and perimeter; 3) Fractal analysis; and 4) Sholl analysis (Green, 2022). Despite advancements in quantitative methods, there remains a lack of comparability between analysis techniques, leading to discrepancies in results (Green, 2022; Reddaway, 2023) and sparking debate over the optimal approach for characterizing microglial morphology.

Moreover, ongoing discussions center around the optimal method for analyzing photomicrographs, with one approach involving the analysis of the entire image and another focusing on single-cell analysis. Green points out potential biases in effect size with full photomicrograph analysis, with studies reporting a significant decrease, ranging from 67-93%, compared to single-cell analysis using the same metrics (Green, 2022). Conversely, despite being more labor-intensive, single-cell analysis often reveals greater statistical differences. However, it may provide a biased sample due to selective cell inclusion. Thus, while full photomicrograph analysis offers efficiency, it is often avoided due to potential reductions in statistical significance. In contrast, single-cell analysis, despite showing larger effect sizes, may represent only a subset of the phenomenon (Green, 2022).

Given the diverse nature of microglia and the unique aspects of each study, it's crucial to identify a sufficient number of variables that capture the complexity of microglial morphology while minimizing the introduction of noise from covariance effects (Siegart, 2022). However, selecting the most appropriate features to characterize microglia for each scenario presents a challenge, as there is not a one-size-fits-all set of features. Acknowledging this challenge and the significant biological variability, we've developed a pipeline that utilizes the four primary methods of microglial analysis, along with machine learning techniques, to pinpoint features that effectively differentiate between study groups. This involves projecting the data into a common space and clustering it to examine the distinct clusters produced by these features. Additionally, since each data point represents a cell morphology, we color-code each cell based on its cluster. This allows for easy visual verification of morphology and patterns detected by the machine learning ensemble. This comprehensive approach aims to highlight the intricacies of microglial morphologies while categorizing them in a clear and unbiased manner.




Morphometrics Analysis

Each cell was first identified in the complete binary photomicrographs, after which classic morphometric features were computed. Skeleton analysis involved measuring the total length of branches (in pixels), counting the number of initial points (also referred to as cell processes emerging from the cell soma), determining the number of junction points (where branches subdivide), and counting the number of endpoints (endpoints of branches). Cell body analysis encompassed calculating the area, perimeter, circularity (where 1 represents a perfect circle), Feret diameter (maximum calliper diameter), compactness (how closely an object packs its area), aspect ratio (width/height), orientation (angle in degrees), and eccentricity (major axis/minor axis). These same features were computed for the entire cell obtained from the image. Fractal analysis involved determining convex hulls (the smallest convex set of pixels enclosing a cell), employing the same calculations as in cell body analysis. Sholl analysis consisted of identifying the number of Sholl circles (circles with increasing radii created around the centroid of the cell soma), counting crossing processes (intersections of cell processes with Sholl circles), and measuring the max distance (maximum distance between the centroid and the four vertices of the image).


![Figure_morphometrics_analysis](https://github.com/Maya-Arteaga/Morphology/assets/70504322/c91266f2-07d2-4a28-85b4-e092a6c8beca)




Feature Selection

Each cell was first identified in the complete binary photomicrographs, after which classic morphometric features were computed. Skeleton analysis involved measuring the total length of branches (in pixels), counting the number of initial points (also referred to as cell processes emerging from the cell soma), determining the number of junction points (where branches subdivide), and counting the number of endpoints (endpoints of branches). Cell body analysis encompassed calculating the area, perimeter, circularity (where 1 represents a perfect circle), Feret diameter (maximum calliper diameter), compactness (how closely an object packs its area), aspect ratio (width/height), orientation (angle in degrees), and eccentricity (major axis/minor axis). These same features were computed for the entire cell obtained from the image. Fractal analysis involved determining convex hulls (the smallest convex set of pixels enclosing a cell), employing the same calculations as in cell body analysis. Sholl analysis consisted of identifying the number of Sholl circles (circles with increasing radii created around the centroid of the cell soma), counting crossing processes (intersections of cell processes with Sholl circles), and measuring the max distance (maximum distance between the centroid and the four vertices of the image).



![Corr_Heatmap](https://github.com/Maya-Arteaga/Morphology/assets/70504322/d2b59693-e12c-4c91-9996-d4efb266614b)





Dimensionality Reduction

Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction technique designed for general non-linear and non-parametric dimensionality reduction while preserving both local and global structures. The algorithm operates under the assumption that the data is uniformly distributed on a Riemannian manifold, where the Riemannian metric remains locally constant and the manifold maintains local connectivity. Note that various hyperparameters (n_neighbors and min_dist) were experimented with, and yet the resulting structure remained robust. (https://github.com/lmcinnes/umap/).


![UMAP_trials](https://github.com/Maya-Arteaga/Morphology/assets/70504322/99d4fcdb-0df3-4339-b2a5-c69c62780d59)




Clustering

Following UMAP, Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) was applied for clustering purposes. HDBSCAN adopts a density-based approach, making minimal implicit assumptions about the clusters. This non-parametric method aims to construct a cluster hierarchy based on the multivariate modes of the underlying distribution. The process involves transforming the space according to density, constructing the cluster hierarchy, and extracting the clusters. Notably, HDBSCAN effectively handles data noise by excluding points that do not fall near densities from the clusters, rendering it more robust to noise and outliers (https://github.com/scikit-learn-contrib/hdbscan).

![UMAP_HDBSCAN_CELLS2](https://github.com/Maya-Arteaga/Morphology/assets/70504322/ccab3478-0ad0-4d50-9b80-f754a08a884b)

![Histogram_Clusters_10_0 01_order](https://github.com/Maya-Arteaga/Morphology/assets/70504322/0d680224-681e-4e12-b9ed-d03076d19c85)

In this case, we are dealing with five groups:

VEH SS: Saline solution as vehicle

ESC: Scopolamine (toxin)

CNEURO 0.1: Treatment at dosage 0.1

CNEURO 1.0: Treatment at dosage 1.0

CNEURO SS: Treatment in a control group

![Figure_pie_chart](https://github.com/Maya-Arteaga/Morphology/assets/70504322/5641aba8-031c-49ac-8915-a2a0c6e28c61)






UMAP projection classified into groups: VEH SS, ESC, CNEURO 0.1, CNEURO 1.0, and CNEURO SS



![UMAP_grupos](https://github.com/Maya-Arteaga/Morphology/assets/70504322/528b9005-be7b-45af-951c-9cae083036c2)







Spatial Visualization

After performing Dimensionality Reduction and Clustering, each cell was color-coded based on its cluster since each data point represents a cell. This approach serves two main purposes: 1) It simplifies the confirmation of similarities among cells belonging to the same cluster in a practical manner; 2) It offers insights into the spatial distribution of each clustered cell. This aspect could prove especially useful for conducting further spatial analyses, considering other structures, and potentially revealing previously unexplored patterns in the physiopathology of diseases.




![Figure_Spatial_Visualization](https://github.com/Maya-Arteaga/Morphology/assets/70504322/ffa50504-7b4d-451d-b38c-6c75bfdea2a4)


Statistical Test

The Chi-square test was performed to investigate the relationship between the six clusters generated and their association with the five study groups. The observed frequency table and the table of standardized residuals are presented. The predominance of clusters 1 and 2 in the diseased group is noted, as well as the absence of groups 4 and 5. In the treated groups, there is a predominance of cluster 4, and clusters 0, 1, and 2 return to values similar to the control group.



![Chi_R](https://github.com/Maya-Arteaga/Morphology/assets/70504322/6dd545d7-2da9-4c23-81cd-fe748a822eb3)

The standardized residuals from the chi-square test reveal important deviations between observed and expected frequencies across clusters and groups in our dataset.

For instance, cluster 1 shows a notable underrepresentation (-3.064) of the VEH_SS group, suggesting fewer observations than expected. Conversely, cluster 1 overrepresents the VEH_ESC group (2.824), indicating a higher-than-expected frequency.

In cluster 4, the CNEURO1_SS group exhibits a significant overrepresentation (2.163), indicating a concentration of observations within this cluster. In contrast, cluster 4 demonstrates a significant underrepresentation of the VEH_ESC group, indicated by a standardized residual of -3.06.

These findings highlight distinct associations and disparities between clusters and groups, providing valuable insights into the dataset's structure and composition.


Juan Pablo Maya Arteaga

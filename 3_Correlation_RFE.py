#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:20:07 2023

@author: juanpablomayaarteaga
"""

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from morpho import set_path
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Record start time
start_time = time.time()


# Load the data
i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
o_path = set_path(i_path + "/Output_images/")
csv_path = set_path(o_path + "/Merged_Data/")
Plot_path = set_path(o_path + "Plots/RFE/")




data = pd.read_csv(csv_path + "Morphology.csv")


# Specify the attributes you want to use in the decision tree
selected_attributes = [
    "Area_soma", "Perimeter_soma", "Circularity_soma", "soma_compactness",
    "soma_feret_diameter", "soma_eccentricity", "soma_aspect_ratio",
    "End_Points", "Junctions", "Branches", "Initial_Points",
    "Total_Branches_Length", "ratio_branches",
    "polygon_area", "polygon_perimeters", "polygon_compactness", "polygon_eccentricities",
    "polygon_feret_diameters",
    "cell_area", "cell_perimeter", "cell_circularity", "cell_compactness", "cell_feret_diameter",
    "cell_eccentricity", "cell_aspect_ratio", "cell_solidity", "cell_convexity",
    "sholl_max_distance", "sholl_crossing_processes", "sholl_num_circles"
]

######################################################################
######################################################################
######################################################################
######################################################################

#######################   CORRELATION MATRIX   ########################
selected_data = data[selected_attributes]

correlation_matrix = selected_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))
# Create a heatmap with a color map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size":8}) #font size of the numbers in the square

# Add a title
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.savefig(Plot_path + "Correlation_Matrix_ALL.png", dpi=800, bbox_inches="tight")
plt.tight_layout()

# Show the plot
plt.show()

######################################################################
######################################################################
######################################################################
######################################################################



######################################################################
######################################################################
######################################################################
######################################################################

#################   RECURSIVE FEATURE ELIMINATION   ##################

from sklearn.model_selection import GridSearchCV

# Specify the target variable
y = data['categories']

# Extract the selected features from the data
X = data[selected_attributes]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)


# Create an SVM classifier for multi-class classification
svc = SVC(kernel="linear", decision_function_shape='ovr')  # 'ovr' stands for One-vs-Rest

# Create the RFE model and select the number of features to retain
num_features_to_keep = X.shape[1] // 2
rfe = RFE(estimator=svc, n_features_to_select=num_features_to_keep)

# Fit the RFE model on the training data
rfe.fit(X_train, y_train)

# Get the ranking of each feature (1 means selected, 0 means not selected)
feature_ranking = rfe.ranking_

# Print the selected features
selected_features = [feature for feature, rank in zip(selected_attributes, feature_ranking) if rank == 1]
print("Selected Features:", selected_features)

# Transform the data to keep only the selected features
X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

# Train a classifier on the selected features
svc.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = svc.predict(X_test_selected)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)
# Additional metrics for multi-class classification
print("Classification Report:\n", classification_report(y_test, y_pred))



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=svc.classes_, yticklabels=svc.classes_)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.savefig(Plot_path + "Confusion_Matrix_SFE.png", dpi=800, bbox_inches="tight")
plt.tight_layout()

plt.show()





# Get the coefficients of the SVM model
svm_coefs = svc.coef_

# Create a DataFrame to store the coefficients and corresponding feature names
coef_df = pd.DataFrame({'Feature': selected_features, 'Coefficient': svm_coefs[0]})

# Sort the DataFrame by coefficient values
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Plot the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
plt.xlabel('Coefficient')
plt.ylabel('Selected Features')
plt.title('Feature Coefficients of SVM Model')
plt.tight_layout()
plt.savefig(Plot_path + "Feature_Coefficients_SVM.png", dpi=800, bbox_inches="tight")
plt.show()




# Record end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")


"""

test size
0.4

Selected Features: ['Circularity_soma', 'soma_compactness', 'soma_eccentricity', 'soma_aspect_ratio', 'Junctions', 'Initial_Points', 'ratio_branches', 'polygon_eccentricities', 'cell_compactness', 'cell_feret_diameter', 'cell_eccentricity', 'cell_aspect_ratio', 'cell_solidity', 'cell_convexity', 'sholl_num_circles']
Selected Features: ['Circularity_soma', 'soma_compactness', 'soma_eccentricity', 'soma_aspect_ratio', 'Junctions', 'Branches', 'Initial_Points', 'ratio_branches', 'polygon_compactness', 'polygon_eccentricities', 'cell_compactness', 'cell_eccentricity', 'cell_aspect_ratio', 'cell_solidity', 'cell_convexity']

Accuracy on Test Set: 0.31825273010920435
Classification Report:
                precision    recall  f1-score   support

CNEURO-01_ESC       0.38      0.55      0.45       150
  CNEURO1_ESC       0.28      0.32      0.30       142
   CNEURO1_SS       0.35      0.22      0.27       129
      VEH_ESC       0.25      0.25      0.25       107
       VEH_SS       0.30      0.19      0.24       113

     accuracy                           0.32       641
    macro avg       0.31      0.31      0.30       641
 weighted avg       0.31      0.32      0.31       641

	Feature	Coefficient
10	cell_eccentricity	0.8179386006460589
13	cell_convexity	0.7991896729751602
1	soma_compactness	0.4333774640695083
11	cell_aspect_ratio	0.2038533391964279
6	ratio_branches	0.12767778857994472
9	cell_feret_diameter	-0.00035894923348678276
14	sholl_num_circles	-0.014856397282073885
5	Initial_Points	-0.03780891495216565
4	Junctions	-0.08533208821972948
12	cell_solidity	-0.22076512545348592
3	soma_aspect_ratio	-0.45477575423589656
7	polygon_eccentricities	-0.48878173317280016
2	soma_eccentricity	-1.099748065524068
8	cell_compactness	-1.754449879373908
0	Circularity_soma	-2.1364472477812626

cross validation score
0.316964
0.366071
0.34375
0.330357
0.401786


svm_coefss:

-2.13645	0.433377	-1.09975	-0.454776	-0.0853321	-0.0378089	0.127678	-0.488782	-1.75445	-0.000358949	0.817939	0.203853	-0.220765	0.79919	-0.0148564
-1.18945	0.503921	-0.373248	-0.623687	-0.114787	-0.181937	-0.141967	-0.244262	-0.931876	-0.00361942	0.355116	0.228098	-0.444918	1.13338	0.148676
-2.11051	0.376814	-1.72632	-0.892463	-0.105422	-0.16669	-0.0673991	0.999017	-0.393119	0.00838825	1.45756	0.355511	-0.418775	0.975492	0.0500068
-0.937797	0.476887	-0.502733	-0.48946	-0.118452	-0.0903641	-0.0209015	0.841382	-1.63942	0.0017708	-0.34554	0.300025	-0.343389	0.601341	0.0930364
0.604716	0.197173	0.0305552	0.0481612	0.000134422	-0.0543415	-0.306618	0.795326	0.312701	-0.00137223	0.331659	0.470737	-0.395343	-0.00678703	0.0833101
-0.826557	0.0442915	-0.302395	-0.217834	0.00864075	-0.0234415	-0.0738965	1.76966	1.26989	0.00787693	0.219221	0.573792	-0.210433	-0.193575	0.0670521
0.952611	0.209039	0.289839	0.0682124	-0.0186303	0.0215272	-0.060893	0.568462	0.172377	0.00117287	-0.084504	0.938409	-0.0420814	-0.500408	0.0990164
-1.00801	-0.147967	-0.909842	-0.407393	0.0238049	0.0960509	0.162111	0.619204	0.651169	0.00490172	-0.0122769	0.114195	0.191024	-0.0992717	0.0141179
0.454071	0.0793532	0.715259	-0.160198	0.00712552	-0.122215	-0.02398	-0.293698	-1.1399	0.0036484	-1.2815	0.248615	0.239141	-0.029777	-0.0305535
1.00968	0.0840735	0.899251	0.435703	-0.0399009	-0.190834	-0.213483	-0.349174	-0.840829	-0.00238031	-1.15499	-0.019907	-0.0140115	0.257592	0.0198972

TIME: 3778seg o 63min





test size
0.3

Selected Features: ['Circularity_soma', 'soma_compactness', 'soma_eccentricity', 'soma_aspect_ratio', 'Junctions', 'Branches', 'Initial_Points', 'ratio_branches', 'polygon_compactness', 'polygon_eccentricities', 'cell_compactness', 'cell_eccentricity', 'cell_aspect_ratio', 'cell_solidity', 'cell_convexity']
Accuracy on Test Set: 0.3367983367983368
Classification Report:
                precision    recall  f1-score   support

CNEURO-01_ESC       0.40      0.59      0.48       113
  CNEURO1_ESC       0.27      0.29      0.28       105
   CNEURO1_SS       0.38      0.28      0.32        99
      VEH_ESC       0.24      0.26      0.25        81
       VEH_SS       0.41      0.19      0.26        83

     accuracy                           0.34       481
    macro avg       0.34      0.32      0.32       481
 weighted avg       0.34      0.34      0.33       481


	Feature	Coefficient
14	cell_convexity	0.8730715967783453
11	cell_eccentricity	0.39802568836682894
1	soma_compactness	0.35299786779705755
7	ratio_branches	0.1670894074172793
12	cell_aspect_ratio	0.14857331258718887
6	Initial_Points	0.07638195646904933
4	Junctions	0.021582250690698856
9	polygon_eccentricities	0.01210341257747416
8	polygon_compactness	0.00196812775448052
5	Branches	-0.06574347705463879
13	cell_solidity	-0.17693790151040645
3	soma_aspect_ratio	-0.2571843108759708
2	soma_eccentricity	-1.0031665209958902
0	Circularity_soma	-1.4763235838289575
10	cell_compactness	-1.8277509819248507




-1.47632	0.352998	-1.00317	-0.257184	0.0215823	-0.0657435	0.076382	0.167089	0.00196813	0.0121034	-1.82775	0.398026	0.148573	-0.176938	0.873072
-0.589538	0.317466	-0.578074	-0.292171	-0.111359	-0.00524471	-0.157683	-0.178813	0.0698032	-0.315853	-0.590897	0.0507837	0.388769	-0.146301	0.860215
-1.50779	0.243436	-1.66648	-0.658803	-0.133624	0.0114782	-0.0826411	-0.0356992	0.0955624	-0.196678	-0.523258	0.642911	0.697368	-0.142152	0.473104
-0.538449	0.238527	-0.414775	-0.4066	0.0151531	-0.0907223	-0.0756002	-0.106005	0.0923109	0.285937	-0.869161	-0.687035	0.559674	0.0989654	0.935581
0.596201	0.184155	-0.0988575	0.317917	-0.000874743	0.000342566	-0.0668563	-0.325236	0.0639192	-0.0898295	0.439411	-0.000994108	0.187395	-0.0591708	-0.408993
-0.951862	-0.011587	-0.101285	-0.028433	-0.160491	0.103722	-0.165806	-0.169679	0.0880104	0.15039	0.909063	-0.806777	0.690979	0.0695243	-0.835653
0.857503	0.28686	0.428537	0.298812	-0.182054	0.108884	-0.312304	-0.382831	0.0498988	0.12721	-0.0606335	-0.936738	0.651878	0.340127	-0.458789
-0.925219	-0.1909	-0.967978	-0.486347	-0.121174	0.0895382	-0.0267329	0.0766471	0.0267013	0.331128	0.490922	-0.265959	0.462827	0.248334	-0.330854
1.04684	0.139495	0.491191	-0.503276	-0.140836	0.0930213	-0.236692	-0.0719989	-0.029326	0.511641	-1.06131	-1.11948	0.615682	0.249642	-0.180985
0.837596	0.0585654	1.19483	0.233427	0.127152	-0.106288	-0.0859146	-0.15821	0.00481389	-0.0513728	-0.351711	-0.753533	-0.0908035	0.0713866	0.539243

Elapsed Time: 4267.341563940048 seconds

"""


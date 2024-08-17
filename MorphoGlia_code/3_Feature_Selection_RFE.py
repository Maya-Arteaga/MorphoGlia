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
import morphoglia as mg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import time

# Record start time
start_time = time.time()


test=0.3

# Load the data
i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro"
o_path = mg.set_path(i_path + "/Output_images/")
csv_path = mg.set_path(o_path + "/Merged_Data/")
Plot_path = mg.set_path(o_path + f"Plots/RFE_{test}/")




data = pd.read_csv(csv_path + "Morphology.csv")


# Get the column names
column_names = data.columns.tolist()

# Starting at the 8th index and filtering out column names containing "_orientation"
variables = column_names[10:]
variables = [col for col in variables if "_orientation" not in col]


print(variables)





######################################################################
######################################################################
######################################################################
######################################################################

#######################   CORRELATION MATRIX   ########################
# Extract the selected data based on the filtered variables
selected_data = data[variables]

# Remove underscores from column names
cleaned_column_names = [col.replace('_', ' ') for col in variables]

# Compute the correlation matrix
correlation_matrix = selected_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create a heatmap with a color map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 8},
            xticklabels=cleaned_column_names, yticklabels=cleaned_column_names)

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


# Specify the target variable
y = data['categories']

# Extract the selected features from the data
X = data[variables]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=24)


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
selected_features = [feature for feature, rank in zip(variables, feature_ranking) if rank == 1]
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



##################### CONFUSION MATRIX #####################


# Remove underscores from class labels
cleaned_classes = [cls.replace('_', ' ') for cls in svc.classes_]

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cleaned_classes, yticklabels=cleaned_classes)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.savefig(Plot_path + f"Confusion_Matrix_SFE_{test}.png", dpi=800, bbox_inches="tight")
plt.tight_layout()

plt.show()

##################### CONFUSION MATRIX #####################

##################### FEATURE IMPORTANCE #####################


# Remove underscores from feature names
cleaned_selected_features = [feature.replace('_', ' ') for feature in selected_features]

# Get the coefficients of the SVM model
svm_coefs = svc.coef_

# Create a DataFrame to store the coefficients and corresponding feature names
coef_df = pd.DataFrame({'Feature': cleaned_selected_features, 'Coefficient': svm_coefs[0]})

# Sort the DataFrame by coefficient values
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

# Plot the coefficients
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='coolwarm')
plt.xlabel('Coefficient')
plt.ylabel('Selected Features')
plt.title('Feature Coefficients of SVM Model')
plt.tight_layout()
plt.savefig(Plot_path + f"Feature_Coefficients_SVM_{test}.png", dpi=800, bbox_inches="tight")
plt.show()




##################### FEATURE IMPORTANCE #####################


######################################################################
######################################################################
######################################################################
######################################################################

##############   CORRELATION MATRIX SELECTED FEATURES  #################


# Filter the column names to remove underscores
cleaned_column_names = [col.replace('_', ' ') for col in selected_features]

# Compute the correlation matrix for the selected features
correlation_matrix_selected = data[selected_features].corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Create a heatmap with a color map
sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', fmt='.2f', 
            annot_kws={"size": 8}, xticklabels=cleaned_column_names, 
            yticklabels=cleaned_column_names)


# Add a title
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.savefig(Plot_path + f"Correlation_Matrix_Selected_{test}.png", dpi=800, bbox_inches="tight")
plt.tight_layout()

# Show the plot
plt.show()

######################################################################
######################################################################
######################################################################
######################################################################

# Record end time
end_time = time.time()

# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")


"""

test size
0.4
 Selected Features: ['Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 'End_Points', 'Junctions', 'Branches', 'Fractal_dimension', 'Convex_Hull_eccentricity', 'Cell_compactness', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_circles', 'Cell_solidity', 'Cell_convexity']
Accuracy on Test Set: 0.47443762781186094
Classification Report:
                precision    recall  f1-score   support

  VEH_SCO_CA1       0.45      0.53      0.48       125
VEH_SCO_HILUS       0.51      0.52      0.51       136
   VEH_SS_CA1       0.56      0.50      0.53       124
 VEH_SS_HILUS       0.37      0.32      0.34       104

     accuracy                           0.47       489
    macro avg       0.47      0.47      0.47       489
 weighted avg       0.47      0.47      0.47       489
 
 
 Elapsed Time: 4710.556609153748 seconds
 Elapsed Time: 599.1945071220398 seconds
 
 ##################################################################
 test size 
 0.3


Selected Features: ['Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 'End_Points', 'Junctions', 'Branches', 'Fractal_dimension', 'Convex_Hull_eccentricity', 'Cell_circularity', 'Cell_compactness', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_circles', 'Cell_convexity']
Accuracy on Test Set: 0.44686648501362397
Classification Report:
                precision    recall  f1-score   support

  VEH_SCO_CA1       0.47      0.53      0.50        97
VEH_SCO_HILUS       0.44      0.46      0.45       103
   VEH_SS_CA1       0.54      0.48      0.51        90
 VEH_SS_HILUS       0.32      0.30      0.31        77

     accuracy                           0.45       367
    macro avg       0.44      0.44      0.44       367
 weighted avg       0.45      0.45      0.45       367



Elapsed Time: 672.8580300807953 seconds



##################################################################

test sixe
0.2




Selected Features: ['Soma_circularity', 'Soma_compactness', 'Soma_eccentricity', 'Soma_aspect_ratio', 'Junctions', 'Branches', 'Initial_Points', 'ratio_branches', 'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Cell_compactness', 'Cell_eccentricity', 'Cell_aspect_ratio', 'Cell_solidity', 'Cell_convexity']
Accuracy on Test Set: 0.3333333333333333
Classification Report:
                precision    recall  f1-score   support

CNEURO-01_SCO       0.40      0.65      0.49        75
CNEURO-10_SCO       0.23      0.27      0.25        64
 CNEURO-10_SS       0.39      0.32      0.35        69
      VEH_SCO       0.27      0.23      0.24        53
       VEH_SS       0.32      0.12      0.17        60

     accuracy                           0.33       321
    macro avg       0.32      0.32      0.30       321
 weighted avg       0.33      0.33      0.31       321


Elapsed Time: 51392.351686000824 seconds

"""


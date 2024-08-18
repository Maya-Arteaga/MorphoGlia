#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 02:54:26 2024

@author: juanpablomaya
"""

#pip install scikit-learn
#pip install seaborn

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import morphoglia as mg

# Record start time
start_time = time.time()

# Load the data
i_path = "/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = mg.set_path(i_path + "/Output_images")
csv_path = mg.set_path(o_path + "/Data/")
prev_path = mg.set_path(o_path + "/Plots/RFE/")

Plot_path = mg.set_path(prev_path + "/Clusters/")

fraction_variables = 2
data = pd.read_csv(csv_path + "Morphology_HDBSCAN_10_0.1_20_10.csv")

# Filter out cluster -1
data = data[data['Clusters'] != -1]

# Specify the attributes you want to use in the decision tree
selected_attributes = ['Soma_area', 'Soma_perimeter', 'Soma_circularity', 
                       'Soma_compactness', 'Soma_orientation', 'Soma_feret_diameter', 
                       'Soma_eccentricity', 'Soma_aspect_ratio', 'End_Points', 
                       'Junctions', 'Branches', 'Initial_Points', 'Total_Branches_Length', 
                       'ratio_branches', 'Convex_Hull_area', 'Convex_Hull_perimeter', 
                       'Convex_Hull_compactness', 'Convex_Hull_eccentricity', 'Fractal_dimension',
                       'Convex_Hull_feret_diameter',
                       'Cell_area', 'Cell_perimeter', 'Cell_circularity', 
                       'Cell_compactness', 'Cell_feret_diameter', 
                       'Cell_eccentricity', 'Cell_aspect_ratio', 'Sholl_max_distance', 
                       'Sholl_crossing_processes', 'Sholl_circles', 'Cell_solidity', 'Cell_convexity']

# Target the Clusters column instead of categories
y = data['Clusters']
X = data[selected_attributes]

test_sizes = [0.4, 0.3, 0.2]

for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=24)

    rf = RandomForestClassifier(random_state=42)
    num_features_to_keep = X.shape[1] // fraction_variables
    rfe = RFE(estimator=rf, n_features_to_select=num_features_to_keep)
    rfe.fit(X_train, y_train)

    selected_features = [feature for feature, rank in zip(selected_attributes, rfe.ranking_) if rank == 1]

    X_train_selected = rfe.transform(X_train)
    X_test_selected = rfe.transform(X_test)

    rf.fit(X_train_selected, y_train)

    y_pred = rf.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix with new category names in bold
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Cluster {int(label)}' for label in np.unique(y)], 
                yticklabels=[f'Cluster {int(label)}' for label in np.unique(y)])
    plt.title(f'Confusion Matrix for test_size={test_size}', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    plt.xticks(rotation=0, fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    plt.savefig(Plot_path + f"Confusion_Matrix_RF_{test_size}_{fraction_variables}.png", dpi=800, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    feature_importances = rf.feature_importances_
    importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Replace underscores with spaces
    importance_df_display = importance_df.copy()
    importance_df_display['Feature'] = importance_df_display['Feature'].str.replace('_', ' ')
    
    # Reverse the Blues palette
    palette = sns.color_palette('Blues', len(importance_df_display))[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df_display, palette=palette)
    plt.xlabel('')
    plt.ylabel('Selected Features', fontweight='bold')
    plt.title(f'Feature Importances for test_size={test_size}', fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.tight_layout()
    plt.savefig(Plot_path + f"Feature_Importances_RF_{test_size}_{fraction_variables}.png", dpi=800, bbox_inches="tight")
    plt.show()

    feature_RFE = importance_df['Feature'].tolist()
    data_RFE = data[feature_RFE]

    correlation_matrix = data_RFE.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size":8}, 
                xticklabels=[label.replace('_', ' ') for label in feature_RFE], 
                yticklabels=[label.replace('_', ' ') for label in feature_RFE])
    plt.title(f'Correlation Matrix Heatmap for test_size={test_size}', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Plot_path + f"Correlation_Matrix_RFE_{test_size}_{fraction_variables}.png", dpi=800, bbox_inches="tight")
    plt.show()
    
    print("________________________________________")
    print("________________________________________")
    print("")
    print(f"Selected Features for test_size={test_size}:", selected_features)
    print("")
    print("________________________________________")
    print("________________________________________")
    print("")
    print(f"Accuracy on Test Set for test_size={test_size}:", accuracy)
    print("")
    print("________________________________________")
    print("________________________________________")
    print("")
    print(f"Classification Report for test_size={test_size}:\n", classification_report(y_test, y_pred))
    print("")
    print("________________________________________")
    print("________________________________________")


# Record end time
end_time = time.time()
elapsed_time = end_time - start_time
elapsed_time_min = elapsed_time / 60

print("")
print(f"Elapsed Time: {elapsed_time} seconds")
print(f"Elapsed Time: {elapsed_time_min} min")
print("")
print("________________________________________")
print("________________________________________")

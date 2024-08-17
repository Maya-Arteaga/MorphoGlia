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
from sklearn.preprocessing import StandardScaler

import time

# Record start time
start_time = time.time()


tests=[0.4, 0.3, 0.2]
#test=4

for test in tests:
    
        
    # Load the data
    i_path = "/Users/juanpablomayaarteaga/Desktop/Hilus/Prepro/"
    o_path = mg.set_path(i_path + "/Output_images/")
    csv_path = mg.set_path(o_path + "/Merged_Data/")
    Plot_path = mg.set_path(o_path + f"Plots/RFE_{test}_scale/")
    
    
    
    
    data = pd.read_csv(csv_path + "Morphology.csv")
    
    
    # Get the column names
    column_names = data.columns.tolist()
    
    # Starting at the 8th index and filtering out column names containing "_orientation"
    variables = column_names[10:]
    variables = [col for col in variables if "_orientation" not in col]
    
    
    print(variables)
    
    # Select the data corresponding to the selected variables
    selected_data = data[variables]
    
    # Scale the selected data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)
    
    # Convert the scaled data back to a DataFrame
    scaled_df = pd.DataFrame(scaled_data, columns=variables)
    
    
    print(scaled_df)
    
    
    ######################################################################
    ######################################################################
    ######################################################################
    ######################################################################
    
    #######################   CORRELATION MATRIX   ########################
    # Extract the selected data based on the filtered variables
    selected_data = scaled_df
    
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
    
    # Extract the selected features from the scaled data
    X = scaled_df[variables]
    
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
    # Visualize the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=svc.classes_, yticklabels=svc.classes_)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(Plot_path + f"Confusion_Matrix_SFE_{test}.png", dpi=800, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
    
    ##################### FEATURE IMPORTANCE #####################
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

  

"""


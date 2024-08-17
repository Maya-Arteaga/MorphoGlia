#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:44:50 2023

@author: juanpablomayaarteaga
"""


import morphoglia as mg
import pandas as pd
import os

#PATHS



i_path = f"/Users/juanpablomaya/Desktop/Hippocampus/Merge/Prepro/"
o_path = mg.set_path(i_path + "/Output_images/")
csv_path= mg.set_path(o_path + "/Merged_Data/")
Plot_path= mg.set_path(o_path+"Plots/")



#VARIABLES

regions= ["HILUS", "CA1"]
subjects = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
groups = ["CNEURO-10", "VEH", "CNEURO-01"]
treatments = ["SCO", "SS"]
tissues = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]



# Empty DF

merged_data = pd.DataFrame()


# Loop to concatenate all the individual DF of  the samples to create one DF 
# which contains the information of all of them 
for region in regions:
    for subject in subjects:
        for group in groups:
            for treatment in treatments:
                for tissue in tissues:
                    #Look in each directory the file "Cell_Morphology.csv"
                    data_directory = f"{subject}_{group}_{treatment}_{tissue}_{region}_FOTO1_PROCESSED/Data/"
                    data_file = "Cell_Morphology.csv"
                    data_path = os.path.join(o_path, data_directory, data_file)
                    
                    if os.path.isfile(data_path):
                        # Read each CSV file and merge them into a single DataFrame
                        df = pd.read_csv(data_path)
                        merged_data = pd.concat([merged_data, df])



# Creating new columns to identify the file origin of the cells and its category

merged_data['Cell_ID'] = ("C"+ merged_data['Cell'].astype(str) + "_" +
                           merged_data['region'] + "_" + 
                           merged_data['group'] + "_" + 
                           merged_data['treatment']+ "_" +
                           merged_data['tissue']+ "_" + 
                           merged_data['subject'])


merged_data['categories'] = (merged_data['group'] + "_" +
                             merged_data['treatment']+ "_" + 
                             merged_data['region'] )

merged_data = merged_data[ ['Cell_ID'] + [col for col in merged_data.columns if col != 'Cell_ID'] ]
merged_data = merged_data[ ['categories'] + [col for col in merged_data.columns if col != 'categories'] ]


# Save to a CSV file
#as Morphology but in a different directory called "Merged_Data". 
#This new Morphology file contains the information of all the micrographs
merged_csv_path = os.path.join(csv_path, "Morphology.csv")
merged_data.to_csv(merged_csv_path, index=False)
print(merged_csv_path)



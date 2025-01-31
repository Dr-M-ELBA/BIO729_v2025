# Lab 3: Comparing the Machine Learning Performance of raw vs Pre-processed Data

In today's lab, we will continue to practice coding a machine learning model and evaluating its performance. However, we will now investigate the effect of pre-processing (e.g., scaling)
on model performance. We will work with three datasets that are all available on sklearn, so there's no data file to download or clone. 

## Objective
To investigate the effect of scaling the data on three models, which are k-nearest neighbour (kNN), support vector machine (SVM) and decision tree (DT)

## Data
We will work with the breast cancer, diabetes and one non-biomedical datasets for predicting house prices. All are accessible via sklearn. I will demonstrate how to access them.

## Instructions
- Import the relevant libraries
- Define your input and output
- Initialise your learners
- Use the holdout method to define your training and testing sets
- Fit the data to your training
- Evaluate the prediction performance
- Repeat the same analysis but scale your data first using the StandardScaler function from sklearn
- Compare the predictive performances between the scaled and unscaled data
- If you have reached this stage, then continue investigating other pre-processing strategies and compare the results to the un-preprocessed data

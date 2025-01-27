# Introduction
Welcome to Lab 2's assignment. Today, we will perform standard machine learning tasks using three different models. This task will help prepare us for the following task (not yet revealed!). We will use the breast-cancer dataset
again and use the three models to predict whether a patient has a malignant or benign tumour. The models will be decision tree (DT), random forest (RF), and k-nearest neighbors (kNN). These are standard machine learning techniques
and are relatively easy to implement using the scikit-learn (sklearn) library. Please see below for further instructions.

Please note, the data has been 'poisoned,' with one column and one row containing corrupted data. Once you've identified them, you can use the following code to clean the data:

```python
# Getting rid off the poison
df_clean = df.drop(["Name_of_Poison_Column"], axis=1)  # Replace "Name_of_Poison_Column" with the actual poisoned column name
df_clean = df_clean.drop([row_index], axis=0)  # Replace row_index with the actual index of the poisoned row
```

Once you have tidied your data, then proceed with the model building. You'll have 30-40 minutes to complete this task

## Objective
To build three machine learning models to predict the diagnosis of a tumour.

## Dataset
The breast-cancer_2.csv dataset. 

## Instructions
You will need to import the correct modules from sklearn, initialise the models, fit the models to your training set (hint: not on the entire dataset), have them predict the outcomes for the test set, and then evaluate to determine
which of the three performed best. To define your training and testing sets, please use the `train_test_split` function, set the `test_size` to 0.3, and set the `random_state` to 42. Feel free to name your variables as you see fit. 
Regarding the evaluation, consider what we discussed during today's lecture.

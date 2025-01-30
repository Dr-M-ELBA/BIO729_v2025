# Lab 4: Visualising Decision Trees, Random Forest, and kNN Exploration

## Introduction
In today's lab, we will continue practising how to build and evaluate decision trees (DT), random forest (RF) and k-nearest neighbour (kNN). We will explore how certain parameter changes can affect their performance. 
We will use the breast cancer data from **scikit-learn**, so there is no need to download or clone any external data file. Remember to separate to your training and testing set.

## Objective
1. **Decision Tree**  
   - Fit a DT and predict the X_test
   - Plot the initial tree and visualise its structure.  
   - On a new line of code, repeat the above but set the `max_depth` parameter to 2, and observe any differences in its shape and performance.  
   - On a new line of code, set the `min_samples_split` parameter to 10 and note how the tree changes.  
   - Report **accuracy** and **MCC** (Matthews Correlation Coefficient) for each variation.

2. **Random Forest**  
   -Fit an RF and predict the X_test
   - select and plot two different trees (of your choosing) from the forest and compare them. The code for selecting one DT from a forest is provided below. 
   - Discuss whether the two trees differ significantly in structure and predictions.

3. **k-Nearest Neighbours (kNN)**  
   - Train a kNN model using different values of `n_neighbors`.  
   - Plot how the performance metric changes with different neighbour values.  
   - Determine the *optimal* neighbour value based on your plots and performance metrics.

## Data
We will use the **Breast Cancer** dataset from **scikit-learn**. This dataset is readily accessible via the `sklearn.datasets` module.

*If you wish, you can also experiment with other datasets, such as the Diabetes or Boston Housing datasets, to see if the trends you observe are consistent across different data.*

## Instructions

1. **Import the relevant libraries**  
   - You will need libraries like `numpy`, `pandas` (optional), and `matplotlib` or `seaborn` for data handling and visualisation.  
   - From `sklearn`, import the dataset, the classifiers (`DecisionTreeClassifier`, `RandomForestClassifier`, `KNeighborsClassifier`), and relevant splitting and metric functions.
   - For visualising trees, you will need the `plot_tree` function from sklearn.tree 

2. **Define your input (features) and output (target)**  
   - Load the data using `load_breast_cancer()`.  
   - Extract your feature matrix `X` and target vector `y`.

3. **Split the data**  
   - Use the holdout method (e.g., `train_test_split`) to define your training and testing sets.  
   - Consider using a fixed `random_state` for reproducibility.

4. **Decision Tree Experiments**  
   1. **Initial Model**  
      - Initialise a `DecisionTreeClassifier`.  
      - Fit the model on your training data.  
      - Plot the tree (e.g., using `plot_tree`).  
      - Predict on the test data and report both **accuracy** and **MCC**.
      - To visualise the tree, you can:
      ```python
         # Plot the decision tree
      plt.figure(figsize=(20, 10))  # Adjust the figure size as desired
      plot_tree(
          clf, # Make sure to use your own DT variable
          filled=True,
          feature_names=data.feature_names,
          class_names=data.target_names
      )
      ```

   2. **Change `max_depth`**  
      - Create a new `DecisionTreeClassifier(max_depth=some_value)`.  
      - Fit and plot the resulting tree.  
      - Compare performance metrics to the initial model.

   3. **Change `min_samples_split`**  
      - Similarly, adjust `min_samples_split` and note how the tree’s structure changes.  
      - Again, compare accuracy and MCC to previous models.

5. **Random Forest Analysis**  
   - Initialise a `RandomForestClassifier` and fit it to your training data.  
   - Compute accuracy and MCC on the test set.  
   - Pick **two specific trees** from the forest (e.g., `forest.estimators_[i]`) and plot them individually with `plot_tree`.  
   - Compare their structures and discuss any differences.
   - The code for selecting one tree and visualising it:
  
    ```python
    # Randomly select a single tree from the forest
    chosen_tree_index = 11
    chosen_tree = rf_clf.estimators_[chosen_tree_index]
    
    print(f"Visualising tree at index: {chosen_tree_index}")
    
    # Plot the chosen tree
    plt.figure(figsize=(20, 10))
    plot_tree(
    chosen_tree,
    filled=True,
    feature_names=data.feature_names,
    class_names=data.target_names
    )
    plt.title(f"Decision Tree from Random Forest (Tree Index: {chosen_tree_index})")
    plt.show()
    ```

6. **kNN Hyperparameter Search**  
   - Initialise a `KNeighborsClassifier` and vary `n_neighbors` within a reasonable range (e.g., 1 to 30).  
   - For each neighbour value, fit the model and record performance (accuracy and/or MCC) on the test set.  
   - Plot the performance as a function of `n_neighbors`.  
   - Identify which neighbour value gives the best result and discuss your findings.
   - Hint: use a for loop to cycle through your `n_neighbors` range

7. **Compare & Conclude**  
   - Summarise how altering `max_depth` or `min_samples_split` in a Decision Tree can change its shape and predictive performance.  
   - Reflect on how individual trees in a Random Forest might differ, even though they are part of the same ensemble.  
   - Identify the optimal `n_neighbors` for kNN and discuss why that might be the case.

## Additional Work
- If you have completed the above steps, try to **tune more hyperparameters** (e.g., `criterion` for Decision Trees, `min_samples_leaf` for Random Forest, or distance metrics for kNN).  
- Consider using **other sklearn datasets** (like Diabetes or Boston Housing) to see if your parameter-tuning insights are dataset-specific or broadly applicable.

---

Feel free to experiment, as the primary goal is to gain a deeper understanding of how each model responds to various hyperparameter settings. Happy coding!

# Task 5 – Decision Trees and Random Forests

This project implements decision tree and random forest classifiers on the Heart Disease dataset.

## Dataset
Used: [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## Steps Covered

1. **Data Preprocessing & EDA**
   - Loaded dataset
   - Generated correlation heatmap
2. **Model Training**
   - Trained a Decision Tree Classifier
   - Controlled overfitting with `max_depth`
   - Trained a Random Forest Classifier
3. **Evaluation**
   - Accuracy and classification reports
   - Cross-validation accuracy
   - Feature importance visualization
4. **Visualization**
   - Decision Tree Plot
   - Feature Importances (Random Forest)

## Results

| Model            | Accuracy (Test) | CV Accuracy |
|------------------|-----------------|-------------|
| Decision Tree    | ~85%            | ~80%        |
| Random Forest    | ~90%            | ~88%        |

## Interview Q&A

- **How does a decision tree work?**  
  It splits the data into branches based on feature thresholds to minimize impurity.

- **What is entropy and information gain?**  
  Entropy measures disorder. Information gain measures the reduction in entropy after a split.

- **Why is Random Forest better?**  
  It reduces overfitting by combining predictions of many trees.

- **What is bagging?**  
  Bootstrap Aggregating: train models on random subsets of data and average predictions.

- **How do you visualize a decision tree?**  
  Using `sklearn.tree.plot_tree()` with appropriate feature/class names.

- **How do you interpret feature importance?**  
  It shows which features most influence predictions, based on split quality.

---

> Run the Python file with:
> ```bash
> python task5_decision_tree_random_forest.py
> ```

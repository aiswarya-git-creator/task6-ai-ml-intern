# Objective
To understand and implement the K-Nearest Neighbors (KNN) algorithm for classification problems using the **Iris dataset**.

# Tools Used
* Python
* Pandas, NumPy
* Matplotlib
* Scikit-learn

# Steps Performed
# Dataset Loading
   * Loaded the Iris dataset from `sklearn.datasets`.
   * Features: Sepal length, Sepal width, Petal length, Petal width.
   * Target: 3 classes of Iris species.
# Feature Normalization
   * Used `StandardScaler` to normalize the features.
# Train-Test Split
   * Dataset split into 80% training and 20% testing using `train_test_split`.
# Model Training and Evaluation
   * Trained `KNeighborsClassifier` for K values from 1 to 10.
   * Calculated accuracy for each K.
   * Printed accuracy results and identified the best K.
# Confusion Matrix
   * Displayed the confusion matrix using `ConfusionMatrixDisplay` for best K.
# Accuracy Visualization
   * Plotted K vs Accuracy using Matplotlib to visualize how accuracy changes with different K values.
# Decision Boundary Visualization
   * Used only the first two features.
   * Visualized decision boundaries using contour plot for the best K.

# Outputs

* Accuracy for Different K Values
* Confusion Matrix for Best K
* KNN Decision Boundary Plot  

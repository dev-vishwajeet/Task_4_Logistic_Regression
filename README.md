# Task 4: Logistic Regression Classification

## Breast Cancer Classification using Logistic Regression

---

## Objective

The objective of this task is to build a Logistic Regression model for breast cancer classification and evaluate its performance using various classification metrics and visualizations.

---

## Tools and Libraries Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

---

## Dataset

The Breast Cancer Wisconsin Dataset was loaded using Scikit-learn's built-in dataset module.

The dataset contains 569 samples and 30 numerical features related to breast cancer cell characteristics.

### Target Classes

* 0 = Malignant (Cancerous)
* 1 = Benign (Non-Cancerous)

---

## Steps Performed

### 1. Dataset Loading

* Loaded the Breast Cancer Dataset using `load_breast_cancer()` from Scikit-learn.
* Converted the feature data into a Pandas DataFrame.

### 2. Dataset Inspection

* Displayed the dataset shape.
* Displayed the first five rows of the dataset using `head()`.

### 3. Train-Test Split

* Split the dataset into training and testing sets using `train_test_split()`.
* Training Data: 80%
* Testing Data: 20%
* Random State: 42

### 4. Feature Scaling

* Standardized all numerical features using `StandardScaler`.
* Applied scaling on training data and transformed testing data using the same scaler.

### 5. Logistic Regression Model

* Created a Logistic Regression model using Scikit-learn.
* Set `max_iter=1000` to ensure proper model convergence.
* Trained the model using the training dataset.

### 6. Prediction

* Generated predictions on the testing dataset using the trained model.

### 7. Classification Report

* Evaluated model performance using:

  * Precision
  * Recall
  * F1-Score
  * Accuracy
* Generated a classification report using `classification_report()`.

### 8. Confusion Matrix

* Generated a Confusion Matrix to compare actual and predicted classifications.
* Visualized the Confusion Matrix using `ConfusionMatrixDisplay`.
* Saved the visualization as:

`screenshots/confusion_matrix.png`

### 9. ROC-AUC Evaluation

* Calculated prediction probabilities using `predict_proba()`.
* Computed ROC-AUC Score using `roc_auc_score()`.
* Generated the ROC Curve using `roc_curve()`.

### 10. ROC Curve Visualization

* Created ROC Curve visualization.
* Plotted:

  * False Positive Rate (FPR)
  * True Positive Rate (TPR)
* Saved the graph as:

`screenshots/roc_curve.png`

---

## Output

* Dataset loaded successfully
* Dataset inspected and analyzed
* Data split into training and testing sets
* Feature scaling completed
* Logistic Regression model trained successfully
* Predictions generated on test data
* Classification Report generated
* Confusion Matrix generated and visualized
* ROC-AUC Score calculated
* ROC Curve generated and saved

---

## Result

Successfully built and evaluated a Logistic Regression model for breast cancer classification. The model was assessed using Classification Report, Confusion Matrix, ROC-AUC Score, and ROC Curve visualization.

# logistic-regression-binary-classifier
This repository implements a binary classifier using logistic reg and covers data preprocessing, feature standardization, model training, and evaluation metrics, including confusion matrix, precision, recall, and ROC-AUC. Visualizations and threshold tuning are included to enhance model performance analysis and understanding of logistic regression.

Cofe for Logistic Regression:

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                            precision_score, recall_score,
                            roc_auc_score, roc_curve,
                            precision_recall_curve, average_precision_score)

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
lr_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_scaled)
y_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
ap_score = average_precision_score(y_test, y_prob)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Average Precision: {ap_score:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, label=f'Logistic Regression (AP = {ap_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.show()

# Feature importance
coefficients = pd.DataFrame({
    'Feature': data.feature_names,
    'Coefficient': lr_model.coef_[0]
}).sort_values('Coefficient', ascending=False)
print("\nTop 10 Features by Coefficient:")
print(coefficients.head(10))

# Threshold analysis
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal Threshold: {optimal_threshold:.4f}")

# Sigmoid function explanation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("\nSigmoid Function Explanation:")
print("The sigmoid function maps real values to probabilities between 0 and 1.")
print("Example: For z=0: σ(0) =", sigmoid(0))
print("         For z=1: σ(1) =", sigmoid(1))

Datasets related to logistic regression binary classification:

1. Breast Cancer Wisconsin Dataset :
      => Contains features computed from a digitized image of a fine needle aspirate of a breast mass. The target variable indicates whether the tumor is malignant (1) or benign (0).

2. Pima Indians Diabetes Dataset :
      => Contains medical data for Pima Indian women, including features like glucose levels, blood pressure, and BMI. The target variable indicates whether the individual has diabetes (1) or not (0).
      
3. Titanic Survival Dataset :
      => Contains data about passengers on the Titanic, including features like age, sex, and class. The target variable indicates whether a passenger survived (1) or not (0).

4. Heart Disease UCI Dataset :
      => Contains various health-related features for patients, with the target variable indicating the presence of heart disease (1) or not (0).
   
5. Bank Marketing Dataset :
       => Contains information about clients of a bank, including features like age, job, and marital status. The target variable indicates whether a client subscribed to a term deposit (1) or not (0).
   
  These datasets are well-suited for practicing logistic regression and binary classification tasks.

      => To perform a binary classification task using logistic regression, follow these steps:

1. Data Collection : Obtain a suitable dataset, such as the Breast Cancer Wisconsin dataset, which contains features and a binary target variable.

2. Data Preprocessing :
   - Cleaning : Handle missing values and remove any irrelevant features.
   - Feature Scaling : Standardize or normalize the features to ensure they are on a similar scale, which helps improve model performance.

3. Splitting the Dataset : Divide the dataset into training and testing sets (commonly a 70-30 or 80-20 split) to evaluate the model's performance on unseen data.

4. Model Training :
   - Import the logistic regression model from a library (e.g., `sklearn`).
   - Fit the model to the training data using the features and the target variable.

5. Model Evaluation :
   - Use the testing set to make predictions.
   - Evaluate the model's performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. A confusion matrix can also provide insights into true positives, false positives, true negatives, and false negatives.

6. Model Tuning : If necessary, adjust hyperparameters or apply techniques like cross-validation to improve model performance.

7. Interpretation : Analyze the model coefficients to understand the impact of each feature on the prediction. This can provide insights into which factors are most influential in determining the outcome.

8. Deployment : Once satisfied with the model's performance, deploy it for real-world predictions, ensuring to monitor its performance over time.

By following these steps, you can effectively implement a logistic regression model for binary classification tasks.

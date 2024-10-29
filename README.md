# Coronary Heart Disease Prediction Model

### Overview
This project focuses on predicting the likelihood of **Coronary Heart Disease (CHD)** using a machine learning approach. The objective is to build a predictive model based on a dataset of medical and lifestyle factors and deploy the model to **Google Cloud Platform (GCP)** using **Vertex AI** with **Airflow** for orchestration. The project involves the following key steps:
- **Exploratory Data Analysis (EDA)**
- **Feature Selection**
- **Ensemble Modeling**
- **Model Development and Evaluation**
- **Model Deployment on GCP using Vertex AI**

The final report and codebase document each step of the process, providing a comprehensive analysis of the model's development and deployment.

---

## Project Structure

---

## Exploratory Data Analysis (EDA)

**EDA** is critical to understanding the dataset and extracting insights to guide model development. The process was broken down into:

### **Univariate Analysis**
- Purpose: To understand the distribution of individual variables.
- Key Steps:
  - Visualizing the distribution of each feature using histograms and box plots.
  - Identifying skewness and outliers in features such as age, cholesterol levels, and blood pressure.
- Key Findings:
  - Some features (e.g., `cholesterol`, `age`, and `BMI`) show strong variability that might influence CHD risk.
  - Missing values were detected in `blood_pressure` and `cholesterol` and were imputed using median values.

### **Bivariate Analysis**
- Purpose: To explore relationships between variables and their potential impact on the target variable (CHD).
- Key Steps:
  - Correlation matrix to assess relationships between features.
  - Visualizing feature-target relationships using scatter plots, box plots, and pair plots.
- Key Findings:
  - Strong correlations were observed between `cholesterol`, `age`, and CHD occurrence.
  - Multicollinearity was handled by removing highly correlated features to avoid redundancy.

For detailed analysis and visualizations, refer to the **EDA Notebook** [here](path_to_EDA_notebook.ipynb).

---

## Feature Selection
Feature selection was performed to improve model performance by focusing on the most predictive variables.

### **Techniques Used**:
1. **Recursive Feature Elimination (RFE)**: Selected top features based on model performance.
2. **Lasso Regularization**: Helped eliminate irrelevant features with zero coefficients.
3. **Correlation-based Filtering**: Removed features with high correlation to reduce multicollinearity.

Key features selected for the final model:
- `age`, `cholesterol`, `blood_pressure`, `smoking_status`, `exercise_frequency`, `family_history`.

For the complete code and process, refer to the **Feature Selection Notebook** [here](path_to_feature_selection_notebook.ipynb).

---

## Ensemble Modeling
We applied **ensemble learning techniques** to enhance the model’s robustness and accuracy.

### **Models Used**:
1. **Random Forest**: A bagging technique that aggregates multiple decision trees.
2. **Gradient Boosting Machines (GBM)**: Applied boosting to sequentially improve the model.
3. **Voting Classifier**: Combined the predictions of Random Forest, Gradient Boosting, and Logistic Regression for better performance.

Key results:
- The **Voting Classifier** yielded the best performance with an accuracy of **85%** and AUC of **0.92**.

---

## Model Development and Training
The final model was developed using the selected features and ensemble techniques. The process involved:

### **Training and Validation**:
- **Train/Test Split**: The dataset was split into 80% training and 20% testing data.
- **Cross-Validation**: 5-fold cross-validation was used to prevent overfitting and ensure model generalization.

### **Model Performance**:
- Evaluation metrics used include **Accuracy**, **Precision**, **Recall**, and **ROC-AUC**.
- The final model had the following performance metrics:
  - **Accuracy**: 85%
  - **Precision**: 83%
  - **Recall**: 80%
  - **AUC**: 0.92

To run the training process:

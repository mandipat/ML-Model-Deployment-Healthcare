# Machine Learning Prediction Pipelines: Breast Cancer and CHD Prediction

## Overview

This repository contains two end-to-end machine learning projects focused on health prediction: **Breast Cancer Prediction** and **Coronary Heart Disease (CHD) Prediction**. Both projects use machine learning pipelines to automate key tasks such as data preprocessing, model training, evaluation, and deployment. The pipelines are deployed on **Google Cloud Platform (GCP)** using **Vertex AI**, with **Apache Airflow** used to automate the workflow.

Each project has a detailed description available in the respective project folders. The pipelines are scalable and ready for production use in healthcare prediction tasks.

---

## Repository Structure

```bash
├── README.md                           # This file
├── breast_cancer_prediction/           # Folder for breast cancer prediction project
│   ├── Breast_Cancer_pipeline.py       # Main script for breast cancer pipeline
│   ├── breast_cancer_pipeline.ipynb    # Jupyter notebooks for experiment tracking
│   └── breast_cancer_README.md         # Detailed description of Breast Cancer project
├── chd_prediction/                     # Folder for coronary heart disease prediction project
│   ├── CHD_Inference pipeline.py       # Main script for CHD prediction pipeline
│   ├── CHD_vertex_ai_deployment.py     # Deployment to GCP Vertex AI
│   ├── CHD_model_train_eval            # Python notebook for Evaluation
│   └── notebooks/                      # Jupyter notebooks for experiment tracking
│       └── Data_Preprocessing.ipynb
│       └── EDA_CHD.ipynb
│   └── chd_README.md                   # Detailed description of CHD prediction project
├── data/                               # Datasets used in both projects
│   ├── breast_cancer_data.csv          # Breast cancer dataset
│   ├── chd_data.csv                    # CHD prediction dataset
```

# Projects Overview

## 1. Breast Cancer Prediction Pipeline
- **Objective**: Predict whether a tumor is benign or malignant based on medical data from a fine needle aspirate (FNA) of breast mass.
- **Model**: Logistic Regression
- **Deployment**: The model is deployed on Google Cloud Vertex AI.
- **Automation**: The pipeline is automated using Apache Airflow.  
For detailed information about this project, including key steps and results, please refer to the [Breast_Cancer_Readme](https://github.com/mandipat/ML-Model-Deployment-Healthcare/blob/main/Breast_Cancer%20Prediction/Breast%20Cancer%20Prediction_README.md)).

## 2. Coronary Heart Disease (CHD) Prediction Pipeline
- **Objective**: Predict the likelihood of developing coronary heart disease within 10 years based on demographic and medical data.
- **Model**: Ensemble methods such as Random Forest and Gradient Boosting Machines.
- **Deployment**: The model is deployed on Google Cloud Vertex AI.
- **Automation**: The pipeline is automated using Apache Airflow.  
For detailed information about this project, including key steps and results, please refer to the [CHD_Readme](https://github.com/mandipat/ML-Model-Deployment-Healthcare/blob/main/CHD_Prediction/CHD%20Prediction_README.md).

# Deployment and Automation with Apache Airflow
Both projects leverage Google Cloud Vertex AI for deploying models in a scalable and production-ready manner. Apache Airflow is used to automate key tasks such as data preprocessing, model training, and deployment.

## Key Automation Features:
- **Data Preprocessing**: Automatically prepare and clean the data for training.
- **Model Training and Evaluation**: Trigger training jobs and automatically evaluate the model’s performance.
- **Model Deployment**: Deploy models to Google Cloud Vertex AI for real-time inference.
- **Scheduling**: Set up regular retraining or evaluation tasks using Airflow’s scheduling capabilities.  
For more details on the Airflow DAGs and how they automate the pipelines, refer to the individual project directories.

# Deployment and Automation with Apache Airflow

Both projects leverage Google Cloud Vertex AI for deploying models in a scalable and production-ready manner. Apache Airflow is used to automate key tasks such as data preprocessing, model training, and deployment.

## Key Automation Features:
- **Data Preprocessing**: Automatically prepare and clean the data for training.
- **Model Training and Evaluation**: Trigger training jobs and automatically evaluate the model’s performance.
- **Model Deployment**: Deploy models to Google Cloud Vertex AI for real-time inference.
- **Scheduling**: Set up regular retraining or evaluation tasks using Airflow’s scheduling capabilities.  

For more details on the Airflow DAGs and how they automate the pipelines, refer to the individual project directories.


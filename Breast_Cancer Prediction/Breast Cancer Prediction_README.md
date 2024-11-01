## "Breast Cancer Prediction Pipeline"
  
  ### Overview: 
  This project focuses on building a **breast cancer prediction model** using **Google Cloud Vertex AI**, 
    following a step-by-step machine learning pipeline. The model is built on the **breast cancer dataset** 
    from **scikit-learn** and uses **logistic regression** for binary classification. 
    The pipeline automates key tasks like data splitting, normalization, model training, evaluation, 
    and deployment with the help of **Google Vertex AI**.

 

  ### Steps in this project:
  
     - step_1:
        title: "Retrieve Breast Cancer Dataset"
        description: |
          The breast cancer dataset is retrieved from **scikit-learn**. 
          It includes features derived from fine needle aspirate (FNA) of breast mass 
          and is used for binary classification (benign or malignant).
    
      - step_2:
        title: "Data Splitting"
        description: |
          The dataset is split into **training** and **validation** partitions to prevent overfitting 
          and ensure proper evaluation of the model. 
          A standard `train_test_split` is applied.
    
    - step_3:
        title: "Data Normalization"
        description: |
          Normalize the dataset to ensure that all features are on the same scale, 
          which helps models like **logistic regression** perform better.
    
    - step_4:
        title: "Logistic Regression Model Training"
        description: |
          The model choice is **Logistic Regression**, a popular and effective model 
          for binary classification problems like breast cancer detection. 
          The model is trained on the normalized training dataset.

    - step_5:
        title: "Model Evaluation"
        description: |
          The model is evaluated using standard classification metrics such as **Accuracy**, 
          **F1-Score**, and **AUC (Area Under the Curve)** to assess how well 
          the model can distinguish between benign and malignant cases.

    - step_6:
        title: "Model Deployment on Google Cloud Vertex AI"
        description: |
          The trained logistic regression model is deployed to **Google Cloud Vertex AI** 
          for scalable and production-ready inference. 
          Steps involve exporting the trained model, uploading it to **Google Cloud Storage** (GCS), 
          and deploying it on **Vertex AI**.

    - step_7:
        title: "Pipeline Automation Using Apache Airflow"
        description: |
          The pipeline tasks (data preprocessing, model training, and deployment) 
          are automated using **Apache Airflow**. A DAG (Directed Acyclic Graph) is created 
          to define tasks and run them sequentially or in parallel as required.


  ### Results:
    Description: 
      The model was evaluated on the validation set using the following metrics:

    metrics:
      - accuracy: "85%"
      - f1_score: "83%"
      - auc: "0.92"

    evaluation_summary: |
      The **Logistic Regression model** achieved an **accuracy of 85%**, with an **F1-Score of 83%**, and an 
      **AUC (Area Under the Curve)** of 0.92, indicating strong predictive performance for detecting whether 
      breast cancer is benign or malignant. The model was able to distinguish between classes with high precision 
      and recall, making it suitable for practical application in medical diagnosis.

  ### Instructions:
  
    - install_dependencies:
        description: "Install required dependencies"
        command: |
          pip install -r requirements.txt

    - run_pipeline:
        description: "Run the data preprocessing and model training"
        command: 
          python Breast_Cancer_pipeline.py

    - deploy_model:
        description: "Deploy the model on Vertex AI"
        command: 
          python src/Breast_Cancer_pipeline.py

    - run_airflow:
        description: 
          If using Airflow, setup Airflow and trigger the DAG from the UI or CLI.

  ### Conclusion: 
  This project demonstrates how to build and automate a machine learning pipeline for breast cancer prediction, 
    from data preprocessing to model deployment. The pipeline is scalable using **Google Cloud Vertex AI**, 
    making it production-ready for real-world applications.


# Integrating-Machine-Learning-with-Real-Time-EHRs-for-the-Diagnosis-of-Cardiovascular-Diseases
Machine learning models for early detection and risk prediction of cardiovascular diseases using Framingham, MIMIC-III, and Cleveland datasets. Achieves up to 99.21% accuracy and 99.89% recall. Designed for real-time EHR integration to support timely, data-driven cardiovascular care decisions.


# AI-Driven Cardiovascular Disease Prognosis and Diagnosis

This repository contains a machine learning project that integrates various models with real-time Electronic Health Record (EHR) systems for the prognosis and diagnosis of cardiovascular diseases (CVD). The project aims to utilize advanced machine learning techniques for early risk prediction, diagnosis, and personalized patient care.

## Project Overview

Cardiovascular diseases (CVD) remain one of the leading causes of global mortality. The early detection of CVD risks is crucial for timely intervention. This project leverages real-time patient data from EHR systems and uses machine learning models to enhance decision-making in clinical settings. The main objective is to apply ML models to data from three well-known datasets to predict cardiovascular risks, improve diagnostic accuracy, and assist clinicians in making informed decisions.

### Machine Learning Models Used
The following models are employed to predict and diagnose CVD:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Neural Network (Multi-Layer Perceptron)**

These models are trained on three publicly available datasets:  
- **Framingham Heart Study**: A cohort study with data on CVD risk factors.  
- **MIMIC-III**: An ICU patient database providing real-time vital signs, medications, and clinical notes.  
- **Cleveland Heart Disease Dataset**: A dataset focused on diagnosing heart disease using features like ECG results and blood pressure levels.

### Key Features
- **Data Preprocessing**: Handling missing data, normalizing features, and encoding categorical variables.
- **Feature Engineering**: Creating interaction features and domain-specific features for improved predictions.
- **Model Selection**: Using various ML algorithms to find the best fit for predicting CVD risk.
- **Evaluation Metrics**: Evaluating models based on accuracy, precision, recall, F1-score, and AUC-ROC.

## Project Structure

```
/data
    └── datasets/                  # Raw datasets (e.g., Framingham, MIMIC-III, Cleveland Heart Disease)
    
/notebooks
    └── CVD_Analysis.ipynb         # Jupyter notebook for ML analysis, data exploration, and model evaluation

/models
    └── trained_models/            # Folder containing trained ML models and checkpoints

/README.md                        # Project documentation
```

## Requirements

To run this project locally, you will need the following dependencies:

- Python 3.7+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Data Preparation

The datasets are publicly available and can be downloaded from the following links:

- [Framingham Heart Study](https://www.framinghamheartstudy.org/)
- [MIMIC-III](https://mimic.mit.edu/)
- [Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

Once downloaded, place them into the `/data/datasets/` folder.

## Running the Project

### Jupyter Notebook

To explore the analysis and models, open the Jupyter Notebook:

1. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Navigate to `CVD_Analysis.ipynb` and run the cells sequentially to preprocess the data, train models, and evaluate their performance.

### Model Training & Evaluation

The following performance metrics are tracked for each model:

- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **AUC-ROC**

These metrics help in evaluating how well the models perform, especially in the context of healthcare, where minimizing false negatives is critical.

### Results

- **Logistic Regression** achieved the highest accuracy (99.21%) with good precision and recall, making it a reliable model for real-time applications.
- **Neural Network** (MLP) showed exceptional recall (99.89%), useful for minimizing false negatives, although its AUC-ROC was lower, which suggests a trade-off between recall and accuracy.
- **Random Forest** and **SVM** also performed well, with Random Forest excelling at handling complex relationships between features.

## Ethical Considerations

When developing AI models for healthcare, ensuring patient data privacy is paramount. This project adheres to HIPAA and GDPR guidelines and uses anonymized data. Additionally, model transparency is essential; therefore, explainable AI techniques (such as SHAP) are employed to provide interpretability.

## Future Work

- **Clinical Deployment**: Deploying these models within real hospital EHR systems.
- **Real-Time Data Integration**: Incorporating streaming data from wearable devices for continuous monitoring.
- **Model Explainability**: Enhancing model transparency and interpretability using SHAP and other explainable AI techniques.

## Conclusion

This project demonstrates the potential of integrating machine learning models with real-time EHR systems for cardiovascular disease prognosis and diagnosis. By utilizing datasets such as Framingham, MIMIC-III, and Cleveland Heart Disease, the models can help in early detection and risk assessment, providing valuable tools for healthcare providers.

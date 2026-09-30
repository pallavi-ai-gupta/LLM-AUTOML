# 🤖 LLM AutoML Framework

## Construction of AutoML Framework Using LLMs

A web-based Automated Machine Learning framework that uses Machine Learning algorithms and a local Large Language Model (LLM) to automatically analyze datasets, train models, compare performance and generate insights.

## 🎯 Objectives

- Upload CSV datasets
- Automatically identify the problem type
- Clean and preprocess data
- Train multiple ML models
- Compare model performance
- Select the best model
- Generate LLM-based analysis
- Visualize dataset and model performance
- Make predictions using the trained model
- Download an automated report

## 🧠 Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Ollama
- Qwen 2.5
- HTML
- CSS

## 🤖 Machine Learning Models

### Classification

- Decision Tree
- Random Forest
- Logistic Regression
- KNN
- SVM

### Regression

- Linear Regression
- Decision Tree
- Random Forest
- KNN
- SVR

## ✨ Main Features

### 1. User Authentication
Users can register, login and logout.

### 2. Dataset Analysis
CSV datasets can be uploaded through the web interface.

### 3. Automatic Data Processing
The framework handles missing values and categorical features.

### 4. AutoML
Multiple machine learning algorithms are automatically trained and compared.

### 5. Model Evaluation
Classification uses:

- Accuracy
- Precision
- Recall
- F1 Score

Regression uses:

- RMSE
- MAE
- R² Score

### 6. LLM Analysis
A local Qwen LLM running through Ollama provides dataset and model insights.

### 7. Visualization
The framework generates:

- Target distribution
- Feature distribution
- Feature correlation
- Model performance charts

### 8. Prediction
The selected trained model can be used to make predictions on new data.

### 9. Report Generation
An automated text report can be downloaded after analysis.

## 📁 Project Structure

```text
LLM-AUTOML/
│
├── app.py
├── Login.html
├── Registration.html
├── users.json
├── README.md
├── .gitignore
│
├── templates/
│   ├── index.html
│   ├── analyze.html
│   ├── result.html
│   └── predict.html
│
├── static/
│
├── saved_model/
│
└── venv/
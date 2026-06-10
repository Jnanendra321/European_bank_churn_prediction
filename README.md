# European_bank_churn_prediction
# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Project Overview

This project focuses on predicting customer churn in the banking sector using Machine Learning techniques. Customer churn prediction helps banks identify customers who are likely to leave their services, enabling proactive retention strategies and reducing customer attrition.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, machine learning model development, model explainability, and deployment through a Streamlit web application.

---

## Problem Statement

Customer retention is a critical challenge for banks. Losing customers can significantly impact profitability and increase acquisition costs. The objective of this project is to build a predictive model capable of identifying customers who are at risk of churning and provide actionable insights for retention strategies.

---

## Objectives

* Predict customer churn accurately.
* Identify factors influencing churn.
* Compare multiple machine learning models.
* Generate customer risk scores.
* Develop an interactive Streamlit dashboard.
* Provide model explainability using Feature Importance and SHAP analysis.

---

## Dataset Information

The dataset contains customer demographic, financial, and behavioral information.

### Features

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Is Active Member
* Estimated Salary

### Target Variable

* Exited (0 = Retained, 1 = Churned)

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* SHAP
* Streamlit
* Joblib

---

## Project Workflow

### 1. Data Collection

* Load customer churn dataset.
* Understand feature distributions and target variable.

### 2. Data Preprocessing

* Handle missing values.
* Encode categorical variables.
* Scale numerical features.
* Address class imbalance using SMOTE.

### 3. Exploratory Data Analysis

* Churn Distribution
* Gender vs Churn
* Geography vs Churn
* Age Analysis
* Balance Analysis
* Correlation Heatmap

### 4. Feature Engineering

* Create meaningful derived features.
* Improve predictive performance.

### 5. Model Development

The following models were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost

### 6. Model Evaluation

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### 7. Explainable AI

* Feature Importance Analysis
* SHAP Summary Plot
* SHAP Bar Plot

### 8. Deployment

The final model is deployed using Streamlit for real-time churn prediction.

---

## Streamlit Dashboard Features

### Home Page

* Project Overview
* Dataset Information
* Key Statistics

### Analytics Dashboard

* Churn Distribution
* Customer Demographics
* Geographic Analysis
* Correlation Analysis

### Prediction Module

* Customer Data Input
* Churn Probability Prediction
* Risk Score Classification

### Explainability Dashboard

* Feature Importance Visualization
* SHAP Analysis

---

## Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 73%      |
| Decision Tree       | 67%      |
| Random Forest       | 82%      |
| XGBoost             | 85%      |

**Best Model:** XGBoost

> Replace 85% with your actual model performance results.

---

## Project Structure

```text
Bank-Customer-Churn-Prediction/
│
├── app.py
├── model.pkl
├── requirements.txt
├── churn.csv
├── Research_Paper.pdf
├── images/
│   ├── dashboard.png
│   ├── feature_importance.png
│   └── shap_plot.png
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Bank-Customer-Churn-Prediction.git
```

Navigate to project directory:

```bash
cd Bank-Customer-Churn-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit application:

```bash
streamlit run app.py
```

---

## Future Enhancements

* Real-time banking integration
* Deep Learning-based churn prediction
* Automated retention recommendation system
* Customer Lifetime Value prediction
* Advanced business intelligence dashboards

---

## Business Impact

This project helps banking institutions:

* Improve customer retention.
* Reduce revenue loss.
* Identify high-risk customers early.
* Support data-driven decision-making.
* Enhance customer engagement strategies.

---

## Author

**Jnanendra Rout**

Data Science | Machine Learning | Predictive Analytics

---

## License

This project is developed for educational and research purposes.

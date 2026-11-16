
# Credit Card Fraud Detection Using Machine Learning

## 1. Project Overview

Credit Card Fraud Detection is a machine learning project developed to identify fraudulent credit card transactions.

The system analyses transaction data and classifies transactions into two categories:

- Genuine Transactions
- Fraudulent Transactions

The project uses machine learning algorithms to detect suspicious transaction patterns and provides a simple interactive web application using Streamlit.

## 2. Objectives

- To analyse credit card transaction data.
- To preprocess and clean the dataset.
- To identify patterns in fraudulent transactions.
- To handle imbalanced data using SMOTE.
- To train machine learning classification models.
- To evaluate and compare model performance.
- To predict fraudulent transactions.
- To develop an interactive fraud detection dashboard.

## 3. Dataset Description

The project uses a credit card transaction dataset named `creditcard.csv`.

The dataset contains the following features:

- Time: Time elapsed between transactions.
- V1 to V28: Anonymized transaction features.
- Amount: Transaction amount.
- Class: Target variable.

Class labels:

- 0 = Genuine transaction
- 1 = Fraudulent transaction

The dataset contains a highly imbalanced distribution of genuine and fraudulent transactions.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| SMOTE | Imbalanced data handling |
| Joblib | Model saving |
| Plotly | Interactive charts |
| Streamlit | Web application |

## 5. Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Duplicate Record Removal
4. Exploratory Data Analysis
5. Feature Scaling
6. Train-Test Splitting
7. Handling Imbalanced Data Using SMOTE
8. Machine Learning Model Training
9. Model Evaluation
10. Fraud Prediction
11. Streamlit Dashboard Development

## 6. Machine Learning Algorithms

### Logistic Regression

Logistic Regression is a supervised classification algorithm used to predict whether a transaction is genuine or fraudulent.

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification performance.

## 7. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

These metrics help evaluate the performance of the fraud detection system.

## 8. Handling Imbalanced Data

The dataset contains significantly more genuine transactions than fraudulent transactions.

SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to generate synthetic minority-class samples.

This helps the machine learning models learn fraudulent transaction patterns.

## 9. Application Features

The Streamlit application includes:

### Dashboard
- Total transaction statistics.
- Genuine and fraudulent transaction counts.
- Fraud percentage.
- Interactive analytical charts.

### Fraud Prediction
- CSV file upload.
- Fraudulent transaction prediction.
- Fraud probability.
- Downloadable prediction results.
- Sample transaction testing.

### Model Performance
- Algorithm comparison.
- Accuracy, precision, recall and F1 score.
- Confusion matrix.
- Model performance visualization.

### About Project
- Project description.
- Technologies used.
- Machine learning workflow.

## 10. Project Folder Structure

```text
Credit_Card_Fraud_Detection/
│
├── app.py
├── train_model.py
├── data_analysis.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── creditcard.csv
│
├── graphs/
│   ├── fraud_distribution.png
│   ├── amount_distribution.png
│   ├── amount_comparison.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── model_comparison.png
│
└── models/
    ├── fraud_model.pkl
    ├── scaler.pkl
    └── model_comparison.csv
```

## 11. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### Step 2: Open Project Folder

```bash
cd Credit_Card_Fraud_Detection
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Models

```bash
python train_model.py
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open at:

http://localhost:8501

## 12. Expected Outcome

The project provides a machine learning-based system capable of classifying credit card transactions as genuine or fraudulent.

It also provides an interactive dashboard for analysing transaction data, comparing machine learning models and testing predictions.

## 13. Future Scope

- Integration with real-time transaction systems.
- Advanced machine learning algorithms.
- Improved fraud detection accuracy.
- Cloud deployment.
- Real-time fraud alerts.
- Continuous model monitoring.

## 14. Conclusion

The Credit Card Fraud Detection project demonstrates how machine learning can be used to identify suspicious financial transactions.

By combining data preprocessing, exploratory data analysis, SMOTE, classification algorithms and an interactive Streamlit dashboard, the project provides an end-to-end demonstration of a fraud detection system.

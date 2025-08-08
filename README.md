🏦 Bank Term Deposit Prediction App
This project is a machine learning web application built with Streamlit that predicts whether a bank customer will subscribe to a term deposit based on their demographic profile and previous campaign interactions. The model is trained on the Bank Marketing dataset and optimized for real-time prediction and usability.

🚀 Features
 User-friendly web interface built with Streamlit

 Predicts whether a customer will subscribe (Yes) or not subscribe (No) to a term deposit

 Displays prediction confidence (probability score)

 Built using a trained XGBoost classifier

 Handles categorical variables through manual label encoding

 Real-time prediction based on user input

 Includes data scaling and preprocessing for robust predictions

🔍 Dataset Information
Dataset: bankmarketing.csv (UCI Bank Marketing Dataset)

Rows: 41,188

Target variable: y (term deposit subscription: yes / no)

Features include:

Age, Job, Marital status, Education

Housing and personal loans

Contact type, Campaign metadata

Economic indicators (e.g., employment rate, euribor)

🤖 Model Information
Model	Accuracy	ROC AUC Score
XGBoost	~87.3%	~0.94
Logistic Regression	~77.5%	~0.89
Gradient Boosting	~86.2%	~0.92

Final Model: XGBoost Classifier

Preprocessing:

Removed outliers and less-informative columns (pdays, poutcome, etc.)

Handled "unknown" categories with mode imputation

Scaled numerical data using MinMaxScaler

Evaluation:

Accuracy

Classification report

Confusion matrix

AUC Score

Precision-Recall Curve

Feature Importance using SHAP and XGBoost importance scores

🛠 Built With
Python 

Streamlit  – for building the interactive UI

XGBoost – powerful gradient boosting model

scikit-learn – for preprocessing, training, and evaluation

SHAP – for interpretability and feature analysis

Pandas – for data manipulation

Matplotlib/Seaborn – for data visualization

Joblib – for model serialization

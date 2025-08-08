## 🏦 Bank Term Deposit Prediction App
This project is a machine learning web application built with Streamlit that predicts whether a bank client will subscribe to a term deposit. It uses demographic, financial, and campaign-related features to deliver predictions with high confidence.

-------------------------------

## ✅ Features
 Interactive web interface built with Streamlit

 Predicts subscription outcome: Yes or No

 Displays probability confidence alongside prediction

 Built using a trained XGBoost classifier

 Handles categorical inputs via manual encoding

 Real-time prediction based on user input

 Trained on over 41,000 records of bank marketing data

---------------------------------

## 🧠 Model Information
Algorithm Used: XGBoost Classifier

Model Selection: Chosen after comparing with Logistic Regression and Gradient Boosting

Final Model Performance:

Accuracy: ~93.8%

ROC AUC Score: ~0.956

Cross-Validation Score (5-fold): [0.938, 0.933, 0.936, 0.932, 0.937]

------------------------------------

## 📊 Dataset Overview
Name: Bank Marketing Dataset (bankmarketing.csv)

Records: 41,188

Target variable: y (Subscribed: yes / no)

Sample Features:

age, job, marital, education, default, housing, loan

contact, month, day_of_week, campaign, previous, poutcome

Economic indicators: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

-----------------------------------------

## 🛠 Built With
Python – Core programming language

Streamlit – Web app framework

XGBoost – Machine learning model

scikit-learn – Preprocessing, metrics, and training utilities

pandas – Data manipulation and exploration

joblib – Model serialization

------------------------------------------

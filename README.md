# Diabetes Prediction using Machine Learning and Neural Networks

## Overview

This project focuses on predicting diabetes in patients using various machine learning algorithms and a custom-built neural network. The primary goal is to compare the performance of different models and identify the most effective approach for accurate diabetes prediction based on patient health data. The project utilizes the Pima Indian Diabetes Dataset from Kaggle.

## Project Description

Diabetes Mellitus is a chronic metabolic disorder affecting the body's ability to use energy from food. Early and accurate diagnosis is crucial for managing the disease and preventing complications. This project explores the application of machine learning and deep learning techniques to predict the presence of diabetes based on health data.

## Technical Approach

The project employs a comparative analysis of several supervised machine learning algorithms, including:

*   Naive Bayes Classifier
*   Gradient Boosting Classifier
*   Logistic Regression
*   K-Nearest Neighbors (KNN)
*   Extra Trees Classifier
*   Voting Ensemble (Logistic Regression, SVM, Decision Tree)

In addition to these algorithms, a custom dense neural network model was developed using Keras and TensorFlow. The neural network architecture consists of:

*   Input layer with 8 nodes (corresponding to the dataset features)
*   Hidden layer with 12 nodes and ReLU activation
*   Output layer with 1 node and Sigmoid activation

Data pre-processing steps include:

*   Handling missing values (imputation with median)
*   Data normalization using StandardScaler
*   Exploratory Data Analysis (EDA) for feature selection and understanding data distribution.

Model performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC AUC.

## Dataset

The Pima Indian Diabetes Dataset was used, which consists of diagnostic measurements from female Pima Indian individuals. The features include:

*   Number of pregnancies
*   Plasma glucose concentration
*   Diastolic blood pressure
*   Triceps skin fold thickness
*   2-Hour serum insulin
*   Body mass index
*   Diabetes pedigree function
*   Age
*   Outcome (diabetes status: 0 or 1)

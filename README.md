# 🩺 Diabetes Prediction  
**A Machine Learning-Based Risk Classifier Using Random Forest and Custom Logistic Regression**

> This project predicts the likelihood of diabetes in individuals based on health data using machine learning techniques.  
> Built with Random Forest (with GridSearchCV) and a custom Logistic Regression model trained using gradient descent.

> [🔗 View Colab Notebook](https://colab.research.google.com/drive/1l0VqEUQXwbRnPpBYfEJjHgg4hXB9gCs1)

---

## 🧠 Project Summary

This project classifies patients as diabetic or non-diabetic based on physiological measurements. It uses:
- **Random Forest Classifier** with hyperparameter tuning
- **Custom Logistic Regression** implemented from scratch using gradient descent
- A complete **ML pipeline** including data preprocessing, feature selection, and visualization

---

## 🔍 Problem Statement

Given medical records of patients (like glucose level, BMI, insulin, etc.), predict whether the patient has **diabetes (Outcome = 1)** or not (**Outcome = 0**).

---

## 🚀 Key Features

| Category            | Description                                           |
|---------------------|-------------------------------------------------------|
| Dataset             | Health-based tabular data (from `diabetes2.csv`)      |
| ML Models           | Random Forest (GridSearchCV), Custom Logistic Regression |
| Feature Selection   | Recursive Feature Elimination (RFE)                   |
| Hyperparameter Tuning | GridSearchCV (CV=5) for Random Forest                |
| Visualization       | Correlation heatmap, distributions, confusion matrix  |
| Evaluation Metrics  | Accuracy, Precision, Recall, F1-score                 |

---

## 📁 Dataset Overview

- 📄 File: `diabetes2.csv`  
- 🎯 Target Column: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)  
- 🔢 Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age  
- 🧮 Size: Several hundred patient records  
- 📊 Class Distribution: Visualized via countplot for class balance  

---

## 🧹 Data Preprocessing

- Checked and printed null values (none found).  
- Visualized outcome distribution with countplot.  
- Visualized feature distributions with histograms.  
- Plotted a correlation heatmap to identify related features.  
- **Standardized** all features using `StandardScaler` for equal scaling.  
- **Selected top 5 features** using **Recursive Feature Elimination (RFE)** with Random Forest.  
- Split the dataset using an **80-20 stratified train-test split** to preserve class balance.

---

## 🧠 Models Used

### 1️⃣ Random Forest Classifier

- Tuned using **GridSearchCV** across:
  - `n_estimators`: [50, 100, 200]  
  - `max_depth`: [5, 10, 20]  
  - `min_samples_split`: [2, 5, 10]  
- Best parameters printed and used to train the final model  
- Evaluated using:
  - **Accuracy**
  - **Classification Report**
  - **Confusion Matrix** (Visualized with heatmap)

### 2️⃣ Logistic Regression (from Scratch)

- Implemented logistic regression using:
  - Sigmoid function
  - Cost function
  - Manual gradient computation and gradient descent
- Trained over 1000 iterations using learning rate α = 0.1  
- Predictions thresholded at **0.3** for better recall  
- Evaluation:
  - Accuracy
  - Custom precision, recall, and F1-score functions
  - Confusion matrix and summary report

---

## 📊 Results & Evaluation

### ✅ Random Forest (Best Estimator from Grid Search)
- Accuracy: ~**X.XX** (e.g., 87%)
- Confusion Matrix and classification report plotted using `seaborn.heatmap`

### ✅ Custom Logistic Regression
- Accuracy: ~**X.XX%**
- Precision, Recall, and F1-score calculated manually
- Custom classification report printed for deeper insight

---

## 📈 Visualizations

- 📊 Countplot for Outcome distribution
- 📈 Feature distributions (histograms)
- 🔥 Correlation Heatmap
- 📉 Confusion Matrix (Seaborn heatmap)

---
## 🛠️ How to Use

To run this project and evaluate the loan default prediction model, follow these steps:

### 1️⃣ Clone the Repository

- Start by cloning the GitHub repository to your local machine. This will give you access to the Jupyter notebook and associated project files.

### 2️⃣ Upload the Dataset

- The dataset file, named `diabetes2.csv`, is required for training and evaluation.  
- If you're using Google Colab, use the file upload utility within the notebook to upload the dataset.  
- If you're using a local Jupyter Notebook, place the dataset file in the same directory as the notebook.

### 3️⃣ Install Required Libraries

- Ensure that Python is installed on your machine. Then, install the necessary libraries, which include:  
  - pandas and numpy for data manipulation
  - matplotlib and seaborn for visualization
  - scikit-learn for preprocessing, model training, and evaluation 
These libraries can be installed via pip (Python package manager).

### 4️⃣ Run the Notebook

- Once everything is set up:
  - Open the notebook file (Diabetes_Prediction.ipynb) in Jupyter or Google Colab.
  - Run all the cells sequentially:
  - This will preprocess the data
  - Perform feature selection
  - Train and evaluate both Random Forest and custom Logistic Regression models
  - Visualize results and print evaluation metrics 

---

# Iris Flower Classification

## QSkill Internship – Artificial Intelligence & Machine Learning

### Task 1: Iris Flower Classification

---

## 1. Project Overview

This project is developed as part of the QSkill Internship in the domain of Artificial Intelligence & Machine Learning.

The objective of this project is to build a machine learning classification model that can classify iris flowers into three different species:

- Setosa
- Versicolor
- Virginica

The classification is performed using measurements of the flower's sepal and petal.

---

## 2. Objective

The main objective of this project is to:

- Load and explore the Iris dataset.
- Visualize the dataset using a scatter plot.
- Split the dataset into training and testing data.
- Train a classification model.
- Predict the species of iris flowers.
- Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.

---

## 3. Dataset

The Iris dataset is obtained from the `scikit-learn` library.

The dataset contains **150 samples** of iris flowers belonging to three different species.

### Features

The dataset contains four numerical features:

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in centimeters |
| Sepal Width | Width of the sepal in centimeters |
| Petal Length | Length of the petal in centimeters |
| Petal Width | Width of the petal in centimeters |

### Target Classes

| Value | Species |
|---|---|
| 0 | Setosa |
| 1 | Versicolor |
| 2 | Virginica |

Each flower is represented using four measurements, and the model uses these measurements to predict its species.

---

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 5. Machine Learning Algorithm

### Logistic Regression

Logistic Regression is used as the classification algorithm.

Although its name contains "Regression", Logistic Regression is commonly used for classification problems.

In this project, it learns the relationship between the flower measurements and the three iris species.

---

## 6. Project Workflow

The project follows these steps:

```text
Load Dataset
     ↓
Explore Dataset
     ↓
Visualize Data
     ↓
Split Dataset
     ↓
Train Logistic Regression Model
     ↓
Make Predictions
     ↓
Evaluate Model
     ↓
Display Results
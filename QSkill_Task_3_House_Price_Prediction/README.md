# House Price Prediction

## QSkill Internship – Artificial Intelligence & Machine Learning

### Task 3: House Price Prediction

---

## 1. Project Overview

This project is developed as part of the QSkill Internship in the domain of Artificial Intelligence & Machine Learning.

The objective of this project is to build a machine learning regression model that predicts the median house value based on different housing-related features.

The project uses the California Housing dataset and Linear Regression for prediction.

---

## 2. Objective

The main objectives of this project are:

- Load and explore a housing dataset.
- Understand the features and target variable.
- Check for missing values.
- Visualize the relationship between income and house value.
- Split the dataset into training and testing data.
- Normalize the input features.
- Train a Linear Regression model.
- Predict house values.
- Evaluate the model using regression metrics.

---

## 3. Dataset

The California Housing dataset is obtained using the `scikit-learn` library.

The dataset contains **20,640 samples** and **9 columns**.

There are:

- 8 input features
- 1 target variable

### Features

| Feature | Description |
|---|---|
| `MedInc` | Median income of the household |
| `HouseAge` | Median age of houses in the block |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Population of the block |
| `AveOccup` | Average number of household members |
| `Latitude` | Latitude of the location |
| `Longitude` | Longitude of the location |

### Target

| Target | Description |
|---|---|
| `MedHouseVal` | Median house value |

The target value is represented in units of **$100,000**.

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

### Linear Regression

Linear Regression is used as the machine learning algorithm.

Linear Regression is a supervised learning algorithm used to predict a continuous numerical value.

In this project, the model learns the relationship between the housing features and the median house value.

---

## 6. Project Workflow

The project follows these steps:

```text
Load Dataset
      ↓
Explore Dataset
      ↓
Check Missing Values
      ↓
Visualize Data
      ↓
Separate Features and Target
      ↓
Split Dataset
      ↓
Normalize Features
      ↓
Train Linear Regression Model
      ↓
Make Predictions
      ↓
Evaluate Model
      ↓
Visualize Predictions
      ↓
Predict New House Value
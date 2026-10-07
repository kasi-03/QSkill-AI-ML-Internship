# QSkill Internship
# Domain: Artificial Intelligence & Machine Learning
# Task 1: Iris Flower Classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. LOAD IRIS DATASET
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("Iris Dataset Loaded Successfully!")


# --------------------------------------------------
# 2. CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["target"] = y

df["species"] = df["target"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})


# --------------------------------------------------
# 3. EXPLORE DATASET
# --------------------------------------------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# --------------------------------------------------
# 4. VISUALIZE DATA
# --------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    hue="species"
)

plt.title("Iris Flower Classification")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")

plt.show()


# --------------------------------------------------
# 5. SPLIT DATA INTO TRAINING AND TESTING
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# --------------------------------------------------
# 6. CREATE AND TRAIN MODEL
# --------------------------------------------------

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# --------------------------------------------------
# 7. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred)


# --------------------------------------------------
# 8. CALCULATE ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# --------------------------------------------------
# 9. CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 11. CONFUSION MATRIX VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


print("\nTask 1 Completed Successfully!")
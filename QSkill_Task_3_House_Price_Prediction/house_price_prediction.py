# QSkill Internship
# Domain: Artificial Intelligence & Machine Learning
# Task 3: House Price Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

housing = fetch_california_housing(as_frame=True)

df = housing.frame

print("California Housing Dataset Loaded Successfully!")


# ============================================================
# 2. EXPLORE DATASET
# ============================================================

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 4. DATA VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="MedInc",
    y="MedHouseVal"
)

plt.title("Median Income vs House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")

plt.tight_layout()
plt.show()


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("MedHouseVal", axis=1)

y = df["MedHouseVal"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print("MedHouseVal")


# ============================================================
# 6. SPLIT DATASET INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================================
# 7. NORMALIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nFeature Normalization Completed!")


# ============================================================
# 8. TRAIN LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# ============================================================
# 9. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")

print(y_pred[:10])


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


print("\nModel Performance")
print("-------------------------")

print("Mean Squared Error:", mse)

print("Root Mean Squared Error:", rmse)

print("Mean Absolute Error:", mae)

print("R2 Score:", r2)


# ============================================================
# 11. ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual House Value")

plt.ylabel("Predicted House Value")

plt.title("Actual vs Predicted House Values")

plt.tight_layout()
plt.show()


# ============================================================
# 12. PREDICT PRICE FOR A NEW HOUSE
# ============================================================

new_house = pd.DataFrame({
    "MedInc": [5.0],
    "HouseAge": [20.0],
    "AveRooms": [6.0],
    "AveBedrms": [1.0],
    "Population": [1000.0],
    "AveOccup": [3.0],
    "Latitude": [34.0],
    "Longitude": [-118.0]
})


# Normalize the new house using
# the SAME scaler used for training data

new_house_scaled = scaler.transform(new_house)


# Predict house value

prediction = model.predict(new_house_scaled)


print("\nPredicted House Value:")

print(prediction[0])

print("\nNote:")
print("House values are represented in units of $100,000.")


# ============================================================
# 13. TASK COMPLETION
# ============================================================

print("\nTask 3 Completed Successfully!")
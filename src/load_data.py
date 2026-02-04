import pandas as pd

# Load the datasets
diabetes = pd.read_csv("data/diabetes.csv")
heart = pd.read_csv("data/heart.csv")

print("Datasets loaded successfully!")
print("Diabetes dataset shape:", diabetes.shape)
print("Heart dataset shape:", heart.shape)

# Preview first 5 rows of each
print("\nDiabetes sample:")
print(diabetes.head())

print("\nHeart sample:")
print(heart.head())

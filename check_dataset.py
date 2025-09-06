import pandas as pd

# Load your dataset
df = pd.read_csv("india_housing_prices.csv")

# Show first 5 rows
print(df.head())

# Show all column names
print("\nColumns in dataset:")
print(df.columns)

# Show basic info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

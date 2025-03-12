
import pandas as pd

# Load the dataset
df = pd.read_csv("data/census.csv")

# Display basic information
print("Dataset Info:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check unique values in categorical columns
print("\nUnique Values in Categorical Columns:")
for col in df.select_dtypes(include=["object"]).columns:
    print(f"{col}: {df[col].nunique()} unique values")
    print(f"Sample values: {df[col].unique()[:5]}")


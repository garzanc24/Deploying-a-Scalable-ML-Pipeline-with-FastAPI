
import pandas as pd

# Load the dataset
df = pd.read_csv("data/census.csv")

# Check for spaces in categorical columns
print("Checking for spaces in categorical columns:")
for col in df.select_dtypes(include=["object"]).columns:
    # Check for leading/trailing spaces
    spaces_count = sum(df[col].str.strip() != df[col])
    print(f"{col}: {spaces_count} values with leading/trailing spaces")
    
    # Display some examples if spaces exist
    if spaces_count > 0:
        examples = df[df[col].str.strip() != df[col]][col].head(3)
        print(f"  Examples: {examples.tolist()}")

# Clean the data by stripping spaces from string columns
print("\nCleaning data...")
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

# Replace "?" with NaN for proper handling
df.replace(" ?", pd.NA, inplace=True)
df.replace("?", pd.NA, inplace=True)

# Check for missing values after replacement
print("\nMissing values after handling ? values:")
print(df.isna().sum())

# Save the cleaned data
df.to_csv("data/clean_census.csv", index=False)
print("\nCleaning complete. Saved to data/clean_census.csv")


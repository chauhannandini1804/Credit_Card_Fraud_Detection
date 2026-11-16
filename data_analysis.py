
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create graphs folder if it does not exist
os.makedirs("graphs", exist_ok=True)

# Set graph style
sns.set_style("whitegrid")

print("Libraries imported successfully!")

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

print("\nDataset loaded successfully!")

# Display first five records
print("\nFirst Five Records:")
print(df.head())

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nTotal Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Check duplicate records
duplicates = df.duplicated().sum()
print("\nTotal Duplicate Records:", duplicates)
# Remove duplicate records
df = df.drop_duplicates().copy()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

print("\nRemaining Duplicate Records:", df.duplicated().sum())

# Transaction distribution
class_counts = df["Class"].value_counts()

print("\nTransaction Distribution:")
print("Genuine Transactions:", class_counts[0])
print("Fraudulent Transactions:", class_counts[1])

# Calculate percentages
percentages = df["Class"].value_counts(normalize=True) * 100

print("\nTransaction Percentages:")
print(percentages)

# Graph 1: Genuine vs Fraudulent Transactions
plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Class")

plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Transaction Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Genuine", "Fraudulent"])

plt.savefig("graphs/fraud_distribution.png")
plt.show()
plt.close()

# Graph 2: Transaction Amount Distribution
plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Amount",
    hue="Class",
    bins=50,
    log_scale=(False, True)
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")

plt.savefig("graphs/amount_distribution.png")
plt.show()
plt.close()

# Graph 3: Correlation Heatmap
plt.figure(figsize=(15, 11))

correlation = df.corr()

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    linewidths=0.1
)

plt.title("Credit Card Transaction Correlation Heatmap")

plt.savefig("graphs/correlation_heatmap.png")
plt.show()
plt.close()

# Graph 4: Transaction Amount Comparison
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Transaction Amount Comparison")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Transaction Amount")

plt.savefig("graphs/amount_comparison.png")
plt.show()
plt.close()

# Final message
print("\n====================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("====================================")

print("\nAll graphs saved inside the graphs folder.")
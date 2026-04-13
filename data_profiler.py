import pandas as pd
import numpy as np

# load dataset
df = pd.read_csv("upi_transactions_messy.csv")

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("Columns:\n", df.columns.tolist())

# ---------------------------------
# Missing values
# ---------------------------------
print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()
print(missing[missing > 0])

# ---------------------------------
# Data types
# ---------------------------------
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# ---------------------------------
# Numeric summary
# ---------------------------------
print("\n========== NUMERIC SUMMARY ==========")
print(df.describe())

# ---------------------------------
# Skew detection
# ---------------------------------
print("\n========== SKEWED FEATURES ==========")
numeric_cols = df.select_dtypes(include=np.number).columns
skew = df[numeric_cols].skew().sort_values(ascending=False)
print(skew)

# ---------------------------------
# Correlation
# ---------------------------------
print("\n========== CORRELATION MATRIX ==========")
corr = df[numeric_cols].corr()
print(corr)
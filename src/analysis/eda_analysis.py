
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("data/processed/user_behavior_features.csv")
print("\nDATASET SHAPE")
print(df.shape)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATA TYPES")
print(df.dtypes)
# missing values
print("\nMISSING VALUES")
print(df.isnull().sum())
print("\nSUMMARY STATISTICS")# numerical
print(df.describe())
# categorical
print("\nCATEGORICAL VALUE COUNTS")
numerical_features = [
    "txn_count",
    "avg_transaction_value",
    "merchant_diversity",
    "weekend_spending_ratio",
    "night_transaction_ratio"
]
# categorical_features = [
#     "favorite_category",
#     "preferred_payment_method",
#     "income_level",
#     "user_profile"
# ]
# for col in categorical_features:
#     print(f"\n{col} VALUE COUNTS")
#     print(df[col].value_counts())   

for col in numerical_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

plt.figure(figsize=(10,6))

corr = df[numerical_features].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")

plt.show()
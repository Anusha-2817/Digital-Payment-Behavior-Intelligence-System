import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# CONFIG

NUM_USERS = 200
NUM_TRANSACTIONS = 4000

merchant_categories = [
    "food", "groceries", "travel", "bills",
    "shopping", "entertainment", "fuel",
    "recharge", "rent", "health"
]

payment_methods = ["UPI", "wallet", "card"]
device_types = ["android", "ios"]
upi_apps = ["GPay", "PhonePe", "Paytm"]

# Generate user base

users = [f"U{str(i).zfill(3)}" for i in range(NUM_USERS)]

data = []

for _ in range(NUM_TRANSACTIONS):

    user = random.choice(users)

# Transaction amount (skewed)
    amount = np.random.exponential(scale=300)

# inject large outliers
    if random.random() < 0.02:
        amount *= random.randint(10, 50)

    amount = round(amount, 2)

# time features
    hour = random.randint(0, 23)
    day = random.randint(0, 6)
    is_weekend = 1 if day in [5, 6] else 0

# merchant category
    category = random.choice(merchant_categories)

# payment method
    payment = random.choice(payment_methods)

# device (with missing values)
    device = random.choice(device_types)
    if random.random() < 0.05:
        device = None

    # balances
    balance_before = np.random.uniform(1000, 50000)
    balance_after = balance_before - amount

# allow negative balance edge case
    if random.random() < 0.01:
        balance_after = -abs(balance_after)

    # transaction gap

    gap = np.random.exponential(scale=60)

    # recurring
    recurring = 1 if random.random() < 0.1 else 0

    # upi app
    upi = random.choice(upi_apps)

    # merchant popularity
    popularity = np.random.uniform(0, 1)

    # network latency
    latency = np.random.normal(300, 150)

# inject weird network spikes
    if random.random() < 0.03:
        latency *= 5

    # failed attempts
    failed = np.random.poisson(0.3)

# inject abnormal failures
    if random.random() < 0.02:
        failed += random.randint(3, 8)

    data.append([
        user, amount, category, payment, hour, day,
        is_weekend, device, balance_before, balance_after,
        gap, recurring, upi, popularity, latency, failed
    ])

columns = [
    "user_id",
    "transaction_amount",
    "merchant_category",
    "payment_method",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "device_type",
    "balance_before",
    "balance_after",
    "transaction_gap_minutes",
    "is_recurring",
    "upi_app",
    "merchant_popularity",
    "network_latency",
    "failed_transaction_attempts"
]

df = pd.DataFrame(data, columns=columns)

# inject additional missing values
for col in ["merchant_category", "payment_method"]:
    df.loc[df.sample(frac=0.03).index, col] = None

df.to_csv("upi_transactions_messy.csv", index=False)

print("Dataset created!")
print(df.head())
print("\nShape:", df.shape)
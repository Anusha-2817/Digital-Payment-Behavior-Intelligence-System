import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

# CONFIG
NUM_USERS = 200

# -------------------------------
# USER PROFILES (NEW)
# -------------------------------

USER_PROFILES = {
    "student": {
        "age_range": (18, 25),
        "income": "low",
        "spend_range": (50, 500),
        "categories": ["food", "entertainment", "recharge"],
        "txn_freq": (30, 80)
    },
    "working": {
        "age_range": (23, 45),
        "income": "medium_high",
        "spend_range": (200, 5000),
        "categories": ["food", "travel", "shopping", "bills"],
        "txn_freq": (20, 60)
    },
    "family": {
        "age_range": (30, 55),
        "income": "medium",
        "spend_range": (300, 4000),
        "categories": ["groceries", "bills", "health"],
        "txn_freq": (15, 40)
    }
}

# -------------------------------
# MERCHANT POOL (NEW)
# -------------------------------

MERCHANTS = [
    {"id": "M101", "category": "food"},
    {"id": "M102", "category": "groceries"},
    {"id": "M103", "category": "travel"},
    {"id": "M104", "category": "bills"},
    {"id": "M105", "category": "shopping"},
    {"id": "M106", "category": "entertainment"},
    {"id": "M107", "category": "fuel"},
    {"id": "M108", "category": "recharge"},
    {"id": "M109", "category": "rent"},
    {"id": "M110", "category": "health"},
]

payment_methods = ["UPI", "wallet", "card"]
device_types = ["android", "ios"]
upi_apps = ["GPay", "PhonePe", "Paytm"]

# -------------------------------
# GENERATE USERS (NEW)
# -------------------------------

users = []

for i in range(NUM_USERS):
    uid = f"U{str(i).zfill(3)}"
    profile = random.choice(list(USER_PROFILES.keys()))
    config = USER_PROFILES[profile]

    user = {
        "user_id": uid,
        "profile": profile,
        "income": config["income"],
        "spend_range": config["spend_range"],
        "preferred_categories": config["categories"],
        "txn_freq": config["txn_freq"]
    }

    users.append(user)

# -------------------------------
# TRANSACTION GENERATION (UPDATED)
# -------------------------------

data = []

for user in users:

    num_txns = random.randint(*user["txn_freq"])

    for _ in range(num_txns):

        # merchant based on preference
        preferred_merchants = [
            m for m in MERCHANTS
            if m["category"] in user["preferred_categories"]
        ]

        merchant = random.choice(preferred_merchants if preferred_merchants else MERCHANTS)

        # amount based on user capacity
        amount = np.random.uniform(*user["spend_range"])

        # rare outliers
        if random.random() < 0.02:
            amount *= random.randint(5, 20)

        amount = round(amount, 2)

        # time features
        hour = random.randint(0, 23)
        day = random.randint(0, 6)
        is_weekend = 1 if day in [5, 6] else 0

        # payment
        payment = random.choice(payment_methods)

        # device
        device = random.choice(device_types)
        if random.random() < 0.05:
            device = None

        # transaction gap
        gap = np.random.exponential(scale=60)

        # recurring (more likely for bills/rent)
        recurring = 1 if merchant["category"] in ["bills", "rent"] and random.random() < 0.4 else 0

        # upi app
        upi = random.choice(upi_apps)

        # merchant popularity
        popularity = np.random.uniform(0, 1)

        # latency
        latency = np.random.normal(300, 150)
        if random.random() < 0.03:
            latency *= 5

        # failed attempts
        failed = np.random.poisson(0.3)
        if random.random() < 0.02:
            failed += random.randint(3, 8)

        data.append([
            user["user_id"],
            merchant["id"],
            amount,
            merchant["category"],
            payment,
            hour,
            day,
            is_weekend,
            device,
            gap,
            recurring,
            upi,
            popularity,
            latency,
            failed,
            user["profile"],
            user["income"]
        ])

# -------------------------------
# DATAFRAME
# -------------------------------

columns = [
    "user_id",
    "merchant_id",
    "transaction_amount",
    "merchant_category",
    "payment_method",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "device_type",
    "transaction_gap_minutes",
    "is_recurring",
    "upi_app",
    "merchant_popularity",
    "network_latency",
    "failed_transaction_attempts",
    "user_profile",
    "income_level"
]

df = pd.DataFrame(data, columns=columns)

# missing values injection
for col in ["merchant_category", "payment_method"]:
    df.loc[df.sample(frac=0.03).index, col] = None

df.to_csv("upi_transactions_behavioral.csv", index=False)

print("Dataset created!")
print(df.head())
print("\nShape:", df.shape)
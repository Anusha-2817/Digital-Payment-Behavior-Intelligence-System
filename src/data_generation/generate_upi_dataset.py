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
        "spend_range": (100, 4000),
        "categories": ["food", "entertainment", "recharge", "shopping"],
        "txn_freq": (25, 90)
    },

    "working": {
        "age_range": (23, 45),
        "income": "medium_high",
        "spend_range": (300, 12000),
        "categories": ["food", "travel", "shopping", "bills"],
        "txn_freq": (15, 70)
    },

    "family": {
        "age_range": (30, 55),
        "income": "medium",
        "spend_range": (200, 9000),
        "categories": ["groceries", "bills", "health", "shopping"],
        "txn_freq": (10, 50)
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
# BEHAVIOR PERSONALITIES (NEW V2)
# -------------------------------

BEHAVIOR_PERSONALITIES = {

    "frugal": {
        "spend_multiplier": 0.90,
        "txn_multiplier": 0.95,
        "night_activity": 0.20
    },

    "balanced": {
        "spend_multiplier": 1.00,
        "txn_multiplier": 1.00,
        "night_activity": 0.30
    },

    "impulsive": {
        "spend_multiplier": 1.15,
        "txn_multiplier": 1.10,
        "night_activity": 0.45
    },

    "luxury": {
        "spend_multiplier": 1.30,
        "txn_multiplier": 0.95,
        "night_activity": 0.35
    },

    "digital_addict": {
        "spend_multiplier": 1.10,
        "txn_multiplier": 1.20,
        "night_activity": 0.50
    }
}
# -------------------------------
# GENERATE USERS (NEW)
# -------------------------------

users = []

for i in range(NUM_USERS):
    uid = f"U{str(i).zfill(3)}"
    profile = random.choice(list(USER_PROFILES.keys()))
    config = USER_PROFILES[profile]
    personality = random.choice(list(BEHAVIOR_PERSONALITIES.keys()))
    personality_config = BEHAVIOR_PERSONALITIES[personality]

    # ---------------------------------
    # PERSONALITY-BEHAVIOR DRIFT
    # ---------------------------------

    drift_probability = 0.35

    # sometimes users behave unlike their archetype
    if random.random() < drift_probability:

        drift_personality = random.choice(
            list(BEHAVIOR_PERSONALITIES.keys())
        )

        personality_config = BEHAVIOR_PERSONALITIES[
            drift_personality
        ]
    user = {
        "user_id": uid,
        "profile": profile,
        "income": config["income"],
        "spend_range": config["spend_range"],
        "preferred_categories": config["categories"],
        "txn_freq": config["txn_freq"],
        "behavior_personality": personality,
        "personality_config": personality_config
    }

    users.append(user)

# -------------------------------
# TRANSACTION GENERATION (UPDATED)
# -------------------------------

data = []

for user in users:

    num_txns = random.randint(*user["txn_freq"])

    for _ in range(num_txns):

        # ---------------------------------
        # REDUCED DETERMINISM
        # ---------------------------------

        random_behavior_probability = 0.25

        # usually follows preferences
        if random.random() > random_behavior_probability:

            preferred_merchants = [
                m for m in MERCHANTS
                if m["category"] in user["preferred_categories"]
            ]

            merchant = random.choice(
                preferred_merchants if preferred_merchants else MERCHANTS
            )

        # sometimes behaves unpredictably
        else:
            merchant = random.choice(MERCHANTS)


        # ---------------------------------
        # BASE AMOUNT
        # ---------------------------------

        amount = np.random.uniform(*user["spend_range"])

        # personality influence
        personality = user["behavior_personality"]
        personality_config = user["personality_config"]

        amount *= personality_config["spend_multiplier"]

        # ---------------------------------
        # CONTRADICTION PROBABILITY
        # ---------------------------------

        contradiction_probability = 0.18

        if random.random() < contradiction_probability:

            # frugal suddenly overspends
            if personality == "frugal":
                amount *= random.uniform(2, 5)

            # luxury user suddenly behaves modestly
            elif personality == "luxury":
                amount *= random.uniform(0.3, 0.7)

            # impulsive users may create chaotic spikes
            elif personality == "impulsive":
                amount *= random.uniform(1.5, 4)

            # balanced users sometimes behave unpredictably
            else:
                amount *= random.uniform(0.5, 2)

        # ---------------------------------
        # RANDOM SPENDING NOISE
        # ---------------------------------

        noise = np.random.normal(1, 0.15)
        amount *= noise

        # rare extreme outliers
        if random.random() < 0.03:
            amount *= random.randint(4, 15)

        amount = round(max(amount, 20), 2)

        # rare outliers
        if random.random() < 0.02:
            amount *= random.randint(5, 20)

        amount = round(amount, 2)

        # time features
        # personality-driven timing
        night_activity = personality_config["night_activity"]

        if random.random() < night_activity:
            hour = random.choice([22, 23, 0, 1, 2, 3])
        else:
            hour = random.randint(6, 21)
        day = random.randint(0, 6)
        is_weekend = 1 if day in [5, 6] else 0

        # payment
        payment = random.choice(payment_methods)

        # device
        device = random.choice(device_types)
        if random.random() < 0.05:
            device = None

        # transaction gap
        base_gap = np.random.exponential(scale=60)

        # impulsive users transact rapidly
        if personality == "impulsive":
            base_gap *= random.uniform(0.3, 0.8)

        # frugal users transact less frequently
        elif personality == "frugal":
            base_gap *= random.uniform(1.5, 3)

        # random noise
        gap_noise = np.random.normal(1, 0.2)

        gap = abs(base_gap * gap_noise)

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
            user["income"],
            user["behavior_personality"]
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
    "income_level",
    "behavior_personality"
]

df = pd.DataFrame(data, columns=columns)

# missing values injection
for col in ["merchant_category", "payment_method"]:
    df.loc[df.sample(frac=0.03).index, col] = None
print("\n=== Personality Distribution ===")
print(df["user_profile"].value_counts())

print("\n=== Transaction Amount Stats ===")
print(df["transaction_amount"].describe())

print("\n=== High Spend Low Income Samples ===")
print(
    df[
        (df["income_level"] == "low") &
        (df["transaction_amount"] > 5000)
    ][
        ["user_profile", "income_level", "transaction_amount"]
    ].head(10)
)
df.to_csv(
    "data/raw/upi_transactions_behavioral.csv",
    index=False
)

print("Dataset created!")
print(df.head())
print("\nShape:", df.shape)

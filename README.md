Synthetic Behavioral Analytics & Archetype Inference Framework

An end-to-end machine learning pipeline for simulating, analyzing, and modeling digital payment behavior using synthetic transaction data.

This project explores how behavioral machine learning systems can become misleading when synthetic datasets are overly deterministic and easily separable. The pipeline was redesigned to introduce realistic ambiguity, overlap, contradiction probability, and behavioral drift in order to better simulate noisy real-world human behavior.

Rather than optimizing purely for accuracy, the project focuses on:

behavioral realism
feature engineering
robustness experimentation
leakage awareness
ambiguity in behavioral inference
synthetic behavior simulation
Project Overview

The system simulates users interacting with a digital payment ecosystem and attempts to infer hidden behavioral archetypes from transaction behavior.

The pipeline combines:

synthetic transaction generation
behavioral feature engineering
archetype inference
behavioral experimentation
model comparison
realism tuning
leakage detection

The project evolved from an initially deterministic system producing artificial 100% accuracy into a more realistic behavioral simulation framework with overlapping user behavior and noisy patterns.

Core Idea

Each simulated user secretly belongs to a hidden behavioral archetype such as:

frugal
balanced
impulsive
luxury
digital_addict

These archetypes influence transaction behavior probabilistically rather than deterministically.

The machine learning models never directly see the archetype label during behavior generation. Instead, they attempt to infer hidden tendencies from noisy behavioral patterns extracted through feature engineering.

Pipeline Architecture
generate_upi_dataset.py
        ↓
Synthetic behavioral transaction generation
        ↓
feat_engg.py
        ↓
Behavioral feature extraction
        ↓
train_classifier.py
        ↓
Archetype inference & model evaluation
File Lifecycle & Data Flow
1. generate_upi_dataset.py
Purpose

Simulates digital payment transactions and user behavior.

Creates
transaction records
spending behavior
temporal behavior
contradictions
behavioral drift
hidden archetypes
Output
data/raw/upi_transactions_behavioral.csv
Output Type

Transaction-level dataset.

Example:

user_id	amount	merchant	hour	behavior_personality

2. feat_engg.py
Purpose

Converts raw transaction logs into behavioral summaries per user.

Input
data/raw/upi_transactions_behavioral.csv
Operations
groups transactions by user
computes behavioral statistics
extracts temporal behavior
computes variance metrics
derives behavioral indicators
Example Engineered Features
txn_count
avg_transaction_value
spending_variance
night_transaction_ratio
category_switch_rate
merchant_diversity
failed_txn_spike_ratio
weekend_night_ratio
Output
data/processed/user_behavior_features.csv
Output Type

One behavioral profile per user.

3. train_classifier.py
Purpose

Trains machine learning models to infer hidden behavioral archetypes.

Input
data/processed/user_behavior_features.csv
Features Used
Basic Features
txn_count
avg_transaction_value
merchant_diversity
weekend_spending_ratio
night_transaction_ratio
Advanced Behavioral Features
spending_variance
high_value_txn_ratio
category_switch_rate
transaction_gap_variance
failed_txn_spike_ratio
weekend_night_ratio
Target Label
behavior_personality
Models Used
Logistic Regression
Random Forest
Gradient Boosting
Output
model comparison
behavioral inference accuracy
robustness comparison

Version Evolution
Version 1 — Deterministic Behavioral System

The original implementation generated highly separable behavioral classes and deterministic pseudo-labels.

Characteristics:

rigid behavior patterns
clustering-generated labels
minimal overlap
deterministic scoring
unrealistic separability
artificial 100% classification accuracy

Problems discovered:

target leakage
pseudo-label leakage
unrealistic feature separability
misleading evaluation metrics
Version 2 — Realistic Behavioral Simulation

The pipeline was redesigned to create more realistic human-like ambiguity.

Enhancements introduced:

overlapping behavioral distributions
contradiction probability
personality-behavior drift
noisy spending patterns
probabilistic behavior generation
reduced deterministic influence
richer behavioral features

Result:

more realistic inference difficulty
reduced artificial separability
improved behavioral realism
lower but more valid accuracy
Experimental Results
Basic Features Only
Model	Accuracy
Logistic Regression	~10%
Random Forest	~22%
Gradient Boosting	~20%

These features alone were insufficient for strong behavioral inference.

Full Behavioral Features
Model	Accuracy
Logistic Regression	~30%
Random Forest	~37%
Gradient Boosting	~40%

The addition of engineered behavioral features significantly improved inference quality.

Earlier deterministic pipeline components were archived after transitioning to the Version 2 probabilistic behavioral simulation architecture

Key Insight

The reduction in accuracy from 100% to ~40% was intentional and reflected improved realism within the synthetic environment.

The project demonstrates that:

highly separable synthetic systems can create misleadingly perfect ML performance
behavioral overlap increases realism
contradictions reduce artificial separability
engineered behavioral features improve inference quality
leakage awareness is critical in behavioral ML systems
Behavioral Features Engineered
Spending Features
average transaction value
spending variance
high-value transaction ratio
total spend
Temporal Features
night transaction ratio
weekend spending ratio
transaction gap variance
late-night transaction count
Merchant Features
merchant diversity
category switching rate
recurring transaction ratio
Risk & Noise Features
failed transaction rate
failed transaction spikes
anomaly-oriented behavioral indicators
Important ML Concepts Explored
feature engineering
behavioral analytics
synthetic data simulation
target leakage
pseudo-label leakage
probabilistic systems
ambiguity in ML
model robustness
overlapping distributions
behavioral drift
explainable behavioral inference
Tech Stack
Python
pandas
NumPy
scikit-learn
matplotlib
Running the Project
Install dependencies
pip install -r requirements.txt
Run full pipeline
python src/data_generation/generate_upi_dataset.py

python src/feature_engineering/feat_engg.py

python src/models/train_classifier.py

Future Improvements
feature importance visualization
SHAP-based explainability
real-world transaction datasets
fraud detection adaptation
behavioral clustering visualization
dashboard integration
real-time behavioral inference
Final Takeaway

This project evolved from a standard synthetic ML classification pipeline into a behavioral simulation and robustness experimentation framework.

The primary focus shifted from maximizing accuracy to understanding:

realism
ambiguity
overlap
behavioral inference difficulty
reliability of evaluation metrics

The resulting system better reflects the complexity and inconsistency of real-world human behavioral data.
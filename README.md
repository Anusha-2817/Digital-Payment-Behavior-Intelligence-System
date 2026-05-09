# UPI & Digital Payment Behavior Intelligence System

🧠 Dataset Behavior Intelligence Pipeline

An end-to-end machine learning pipeline for analyzing tabular datasets, comparing model behavior, identifying hidden patterns, and generating interpretable insights.

The project focuses on behavior-aware data analysis rather than only maximizing prediction scores.

Overview

This project was built to explore how ML pipelines can move beyond basic prediction tasks and provide meaningful understanding of structured data.
The system:

profiles datasets automatically
compares multiple ML models
analyzes feature importance
detects hidden behavioral patterns
performs clustering-based segmentation
generates interpretable reports

The current implementation uses a synthetic digital payment transaction dataset to simulate real-world behavioral analytics scenarios.

Pipeline Architecture:
Load Dataset
↓
Data Profiling & Cleaning
↓
Feature Engineering
↓
Categorical Encoding
↓
Model Training & Comparison
↓
Evaluation & Validation
↓
Feature Importance Analysis
↓
Behavioral Clustering
↓
Interpretability & Final Report

📊 Current Capabilities
Dataset Profiling
Detects dataset structure
Identifies numerical/categorical columns
Handles missing values and noisy inputs
Feature Engineering
Removes irrelevant identifiers
Detects potential leakage features
Encodes categorical variables
Builds behavioral indicators
Machine Learning

Models currently explored:

Linear Regression
Random Forest Regressor
Gradient Boosting Regressor

Evaluation metrics:

R² Score
MAE
RMSE
Clustering & Segmentation
KMeans-based behavioral grouping
User/spending segmentation
Pattern discovery across transaction behaviors
Interpretability
Feature importance ranking
Model comparison reporting
Behavioral trend analysis

# 📌 Example Report

# DATASET ANALYSIS REPORT

Samples: 4000
Features: 16

Problem Type: Regression
Dataset Type: Non-linear

Best Model: RandomForest
Performance:

- R²: 0.291
- MAE: 468.06
- RMSE: 1737.6

Top Features:

- balance_before
- merchant_popularity
- transaction_gap_minutes

Clusters Identified:

- Cluster 0 → Regular Users
- Cluster 1 → Low Activity Users
- Cluster 2 → High Spending Users

Observation:
Tree-based models performed better due to non-linear feature interactions.

🧠 Concepts Explored
Feature Engineering
Data Leakage Detection
Regression Modeling
Behavioral Segmentation
Feature Importance Analysis
Clustering Techniques
Explainable Machine Learning
Synthetic Data Simulation

⚠️ Important Learning

One major focus of this project is understanding how misleading performance metrics can occur due to feature leakage.
Example:
balance_after
was initially included as a feature, which unintentionally leaked future transaction information into the model.
Removing leakage-related features reduced model performance slightly but improved the validity and reliability of the results.

▶️ Running the Project
pip install -r requirements.txt
python main.py

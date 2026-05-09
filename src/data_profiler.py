import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def dataset_overview(df):
    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }

def feature_types(df):
    numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    
    return numeric, categorical

def missing_values(df):
    return df.isnull().sum()

def data_types(df):
    return df.dtypes

def numeric_summary(df):
    return df.describe()

def skewness(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    return df[numeric_cols].skew().sort_values(ascending=False)

def correlation(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    return df[numeric_cols].corr()
def detect_problem_type(df):
    if df.select_dtypes(include=np.number).shape[1] > 0:
        return "Likely Regression"
    else:
        return "Likely Classification"
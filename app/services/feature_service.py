import pandas as pd

from src.feature_engineering.feat_engg import generate_behavior_features

def process_transactions(df):

    features = generate_behavior_features(df)

    return features
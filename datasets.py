import pandas as pd

def ml_100k():
    return pd.read_csv('data/ml-100k/u.data', '\t', names=['user', 'item', 'rating', 'timestamp'])
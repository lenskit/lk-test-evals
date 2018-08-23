import pandas as pd

def ml_100k():
    return pd.read_csv('data/ml-100k/u.data', '\t', names=['user', 'item', 'rating', 'timestamp'])

def ml_1m():
    return pd.read_csv('data/ml-1m/ratings.dat', '::', names=['user', 'item', 'rating', 'timestamp'])
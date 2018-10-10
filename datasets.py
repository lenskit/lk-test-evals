import pandas as pd

def ml_100k():
    return pd.read_csv('data/ml-100k/u.data', sep='\t',
                       names=['user', 'item', 'rating', 'timestamp'])

def ml_1m():
    return pd.read_csv('data/ml-1m/ratings.dat', sep='::',
                       names=['user', 'item', 'rating', 'timestamp'])

def ml_10m():
    return pd.read_csv('data/ml-10M100K/ratings.dat', sep='::',
                       names=['user', 'item', 'rating', 'timestamp'])

def ml_20m():
    df = pd.read_csv('data/ml-20m/ratings.csv')
    return df.rename(columns={'movieId': 'item', 'userId': 'user'})

def ml_small():
    df = pd.read_csv('data/ml-latest-small/ratings.csv')
    return df.rename(columns={'movieId': 'item', 'userId': 'user'})
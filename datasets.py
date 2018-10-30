from pathlib import Path
import logging
import pandas as pd

_log = logging.getLogger('exp.datasets')

def current(src, dst):
    if not dst.exists():
        return False
    if src.stat().st_mtime > dst.stat().st_mtime:
        return False
    
    return True


def ml_100k():
    data = pd.read_csv('data/ml-100k/u.data', sep='\t',
                        names=['user', 'item', 'rating', 'timestamp'])
    data['rating'] = data.rating.astype('f8')
    return data

def ml_1m():
    data = Path('data/ml-1m')
    raw = data / 'ratings.dat'
    pq = data / 'ratings.parq'
    if current(raw, pq):
        return pd.read_parquet(pq)
    else:
        data = pd.read_csv(raw, sep='::',
                        names=['user', 'item', 'rating', 'timestamp'])
        data['rating'] = data.rating.astype('f8')
        _log.info('saving %s', pq)
        data.to_parquet(pq)
        return data

def ml_10m():
    data = Path('data/ml-10M100K')
    raw = data / 'ratings.dat'
    pq = data / 'ratings.parq'
    if current(raw, pq):
        return pd.read_parquet(pq)
    else:
        data = pd.read_csv(raw, sep='::',
                           names=['user', 'item', 'rating', 'timestamp'])
        _log.info('saving %s', pq)
        data.to_parquet(pq)
        return data

def ml_20m():
    data = Path('data/ml-20m')
    raw = data / 'ratings.csv'
    pq = data / 'ratings.parq'
    if current(raw, pq):
        return pd.read_parquet(pq)
    else:
        df = pd.read_csv(raw)
        df = df.rename(columns={'movieId': 'item', 'userId': 'user'})
        _log.info('waving %s', pq)
        df.to_parquet(pq)
        return df

def ml_small():
    df = pd.read_csv('data/ml-latest-small/ratings.csv')
    return df.rename(columns={'movieId': 'item', 'userId': 'user'})

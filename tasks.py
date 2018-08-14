import os.path
from pathlib import Path
from invoke import task
import logging

import numpy as np
import pandas as pd

from lenskit import batch

logging.basicConfig(level='INFO', format='{levelname} {name} {message}', style='{')

_log = logging.getLogger('lk-test-evals')

@task
def ensure_directories(c):
    build = Path('build')
    build.mkdir(exist_ok=True)

    (build / 'lenskit').mkdir(exist_ok=True)
    (build / 'lkpy').mkdir(exist_ok=True)


@task(ensure_directories)
def train_lenskit(c, algorithm='item-item', data='ml-100k', output=None):
    if output is None:
        output = 'build/lenskit/{}-{}.model'.format(algorithm, data)

    lk = Path('lenskit/bin/lenskit')
    
    c.run('{} train-model --data-source data/{}.yml -o {} algorithms/{}.groovy'.format(lk, data, output, algorithm))


@task(ensure_directories)
def train_lkpy(c, algorithm='item-item', data='ml-100k', output=None):
    if output is None:
        output = 'build/lkpy/{}-{}.hdf'.format(algorithm, data)
    
    import algorithms
    import datasets

    a = getattr(algorithms, algorithm.replace('-', '_'))
    ds = getattr(datasets, data.replace('-', '_'))
    ds = ds()
    _log.info('training %s on %s', a, data)
    model = a.train(ds)
    _log.info('saving model to %s', output)
    a.save_model(model, output)


@task(ensure_directories)
def sample_users(c, data='ml-100k', nusers=100, nitems=20, force=False):
    file = 'build/pairs-{}.csv'.format(data)
    if os.path.exists(file) and not force:
        _log.info('%s exists, skipping', file)
        return

    import datasets
    ds = getattr(datasets, data.replace('-', '_'))
    ds = ds()

    items = ds.item.unique()
    users = ds.item.unique()

    users = np.random.choice(users, nusers, replace=False)
    _log.info('samping for %d users', len(users))
    pairs = (pd.DataFrame({'user': u, 'items': np.random.choice(items, nitems, replace=False)})
             for u in users)
    pairs = pd.concat(pairs)
    _log.info('writing sample to %s', file)
    pairs.to_csv(file, header=False, index=False)


@task(pre=[sample_users, train_lenskit])
def predict_lenskit(c, algorithm='item-item', data='ml-100k', model=None, output=None):
    if model is None:
        model = 'build/lenskit/{}-{}.model'.format(algorithm, data)
    if output is None:
        output = 'build/lenskit/{}-{}-preds.csv'.format(algorithm, data)
    pair_file = 'build/pairs-{}.csv'.format(data)
    
    lk = Path('lenskit/bin/lenskit')
    c.run('{} predict --data-source data/{}.yml -m {} -B {} -o {}'.format(lk, data, model, pair_file, output))


@task(pre=[sample_users, train_lkpy])
def predict_lkpy(c, algorithm='item-item', data='ml-100k', model=None, output=None):
    if model is None:
        model = 'build/lkpy/{}-{}.hdf'.format(algorithm, data)
    if output is None:
        output = 'build/lkpy/{}-{}-preds.csv'.format(algorithm, data)
    pair_file = 'build/pairs-{}.csv'.format(data)

    import algorithms
    
    a = getattr(algorithms, algorithm.replace('-', '_'))
    _log.info('loading model from %s', model)
    mod = a.load_model(model)

    pairs = pd.read_csv(pair_file, names=['user', 'item'])
    _log.info('predicting for %d pairs', len(pairs))
    preds = batch.predict(a, pairs, model=mod)
    preds.to_csv(output, index=False)


if __name__ == '__main__':
    import invoke.program
    program = invoke.program.Program()
    program.run()

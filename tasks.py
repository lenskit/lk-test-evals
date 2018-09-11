import os.path
from pathlib import Path
from invoke import task
import logging, logging.config
import yaml

import numpy as np
import pandas as pd

from lenskit import batch

with open('logging.yaml') as lf:
    log_config = yaml.load(lf)

logging.config.dictConfig(log_config)

_log = logging.getLogger('exp.runner')

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
    log = 'build/lenskit/{}-{}.log'.format(algorithm, data)
    
    c.run('{} --log-file {} --log-file-level=TRACE train-model --data-source data/{}.yml -o {} algorithms/{}.groovy'.format(lk, log, data, output, algorithm))


@task(ensure_directories)
def train_lkpy(c, algorithm='item-item', data='ml-100k', output=None, debug=False):
    if output is None:
        output = 'build/lkpy/{}-{}.hdf'.format(algorithm, data)
    
    import algorithms
    import datasets

    if debug:
        logging.getLogger('lenskit').setLevel('DEBUG')

    a = getattr(algorithms, algorithm.replace('-', '_'))
    ds = getattr(datasets, data.replace('-', '_'))
    ds = ds()
    _log.info('training %s on %s with %d rows', a, data, len(ds))
    model = a.train(ds)
    _log.info('saving model to %s', output)
    a.save_model(model, output)
    logging.getLogger('lenskit').setLevel('INFO')


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


@task
def predict_lenskit(c, algorithm='item-item', data='ml-100k', model=None, output=None):
    if model is None:
        model = 'build/lenskit/{}-{}.model'.format(algorithm, data)
    if output is None:
        output = 'build/lenskit/{}-{}-preds.csv'.format(algorithm, data)
    pair_file = 'build/pairs-{}.csv'.format(data)
    log = 'build/lenskit/{}-{}-predict.log'.format(algorithm, data)
    
    lk = Path('lenskit/bin/lenskit')
    c.run('{} --log-file={} --log-file-level=DEBUG predict --data-source data/{}.yml -m {} -B {} -o {}'.format(lk, log, data, model, pair_file, output))


@task
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
    _log.info('writing predictions to %s', output)
    preds.to_csv(output, index=False)


@task
def time_train(c, algorithm='item-item', data='ml-100k', type='openmp', threads=None, output='build/timing.csv', n=10, debug=False):
    import ii_scaling

    import algorithms
    import datasets

    if debug:
        logging.getLogger('lenskit').setLevel('DEBUG')

    a = getattr(algorithms, algorithm.replace('-', '_'))
    ds = getattr(datasets, data.replace('-', '_'))
    ds = ds()

    _log.info('timing training of %s on %s with %d rows', a, data, len(ds))
    ii_scaling.test_and_run(a, ds, output, n, dataset=data, mptype=type, threads=threads)


if __name__ == '__main__':
    import invoke.program
    program = invoke.program.Program()
    program.run()

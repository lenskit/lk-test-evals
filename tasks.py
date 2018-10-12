import os.path
import time
from pathlib import Path
from invoke import task
import logging, logging.config
import yaml
import sqlite3

import numpy as np
import pandas as pd
import numba

from lenskit import batch, util

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
    start = time.perf_counter()
    _log.info('training %s on %s with %d rows', a, data, len(ds))
    model = a.train(ds)
    elapsed = time.perf_counter() - start
    _log.info('trained %s on %s in %.2fs', a, data, elapsed)
    _log.info('saving model to %s', output)
    a.save_model(model, output)
    logging.getLogger('lenskit').setLevel('INFO')
    try:
        _log.info('used threading layer %s', numba.threading_layer())
    except ValueError:
        _log.info('did not use Numba multithreading')


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
def time_train(c, algorithm='item-item', data='ml-100k', type='openmp',
               threads=None, mkl_threads=None, output='build/timing.csv', n=10, debug=False):
    import timing

    import algorithms
    import datasets

    if debug:
        logging.getLogger('lenskit').setLevel('DEBUG')

    a = getattr(algorithms, algorithm.replace('-', '_'))
    ds = getattr(datasets, data.replace('-', '_'))
    ds = ds()

    _log.info('timing training of %s on %s with %d rows', a, data, len(ds))
    timing.test_and_run(a, ds, output, n, dataset=data, mptype=type,
                        threads=threads, mkl_threads=mkl_threads)


@task
def sweep(c, algorithm='item-item', data='ml-100k'):
    import datasets
    import sweeps

    timer = util.Stopwatch()

    ds = getattr(datasets, data.replace('-', '_'))
    sf = getattr(sweeps, 'sweep_' + algorithm.replace('-', '_'))

    _log.info('sweeping %s on %s', algorithm, data)
    fn = 'build/sweep-{}-{}'.format(algorithm, data)
    path = Path(fn)
    _log.info('saving results to %s', fn)
    sf(ds(), fn)
    _log.info('finished sweep in %s', timer)


@task
def probe(c, experiment, data='ml-100k'):
    import datasets
    import probes

    ds = getattr(datasets, data.replace('-', '_'))
    sf = getattr(probes, 'probe_' + experiment.replace('-', '_'))

    _log.info('probing %s on %s', experiment, data)
    fn = 'build/probe-{}-{}.csv'.format(experiment, data)
    res = sf(ds())
    _log.info('finished probe, saving to %s', fn)
    res.to_csv(fn, index=False)


if __name__ == '__main__':
    import invoke.program
    program = invoke.program.Program()
    program.run()

import os
import time
import logging
import numpy as np
import pandas as pd

_log = logging.getLogger('exp.scaling')

def time_train(algo, ratings):
    start = time.perf_counter()
    _log.debug('training %s', algo)
    model = algo.train(ratings)
    end = time.perf_counter()
    elapsed = end - start
    _log.info('trained %s in %.3fs', algo, elapsed)
    return elapsed


def time_n_trains(algo, ratings, n=10):
    times = np.zeros(n)
    for i in range(n):
        times[i] = time_train(algo, ratings)
    return times

def test_and_run(algo, ratings, file='timing.csv', n=10, **kwargs):
    past = None
    if os.path.exists(file):
        past = pd.read_csv(file)

    cols = dict(kwargs)
    cols.update({'run': np.arange(n),
                 'time': time_n_trains(algo, ratings, n)})
    frame = pd.DataFrame(cols)
    if past is not None:
        frame = pd.concat([past, frame])
    frame.to_csv(file, index=False)

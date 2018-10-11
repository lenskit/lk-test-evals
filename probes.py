import logging

import pandas as pd
import numpy as np

from lenskit import batch, util, crossfold as xf
from lenskit.algorithms import als

_log = logging.getLogger('exp.probes')

def probe_als(data, max_iters=50):
    algo = als.BiasedMF(60, max_iters)
    (train, test), = xf.sample_users(data, 1, 250, xf.SampleN(5))
    runs = []

    algo.timer = util.Stopwatch()

    init, uctx, ictx = algo._initial_model(train)

    for epoch, model in enumerate(algo._train_iters(init, uctx, ictx)):
        preds = batch.predict(algo, test, model)
        rmse = np.sqrt(np.sum(np.square(preds.rating - preds.prediction)) / len(preds))
        _log.info('epoch %d has RMSE %.3f', epoch, rmse)
        runs.append({'epoch': epoch, 'rmse': rmse})
    
    return pd.DataFrame(runs)
import time
import logging

import pandas as pd

from lenskit import batch
import lenskit.crossfold as xf
from lenskit.algorithms import als, funksvd, item_knn

_log = logging.getLogger('exp.sweep')


def sweep(dbc, data, instances, fields):
    "Sweep over a set of instances using data."

    for part, (train, test) in enumerate(xf.partition_users(data, 5, xf.SampleN(5))):
        for algo in instances:
            _log.info('training %s on partition %d', algo, part)
            start_time = time.perf_counter()
            model = algo.train(train)
            train_time = time.perf_counter()
            _log.info('trained model in %.2fs', train_time - start_time)
            preds = batch.predict(algo, test, model)
            pred_time = time.perf_counter()
            _log.info('computed predictions in %.2fs', pred_time - train_time)
            run = {'Partition': part, 'TrainTime': train_time - start_time, 'PredictTime': pred_time - train_time}
            for f in fields:
                run[f] = getattr(algo, f)
                preds[f] = getattr(algo, f)
            pd.DataFrame([run]).to_sql('runs', dbc, if_exists='append')
            preds.to_sql('predictions', dbc, if_exists='append')
            dbc.commit()


def sweep_als(data, dbc):
    "Sweep the ALS MF algorithm."
    sizes = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    regs = [0.01, 0.05, 0.1, 0.2]
    instances = [als.BiasedMF(sz, iterations=15, reg=reg)
                 for sz in sizes
                 for reg in regs]
    
    sweep(dbc, data, instances, ['features', 'regularization'])

    
def sweep_item_item(data, dbc):
    "Sweep the item-item k-NN algorithm."
    sizes = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    instances = [item_knn.ItemItem(n) for n in sizes]

    sweep(dbc, data, instances, ['max_neighbors'])
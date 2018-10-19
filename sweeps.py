import time
import logging

import pandas as pd
import fastparquet as fpq

from lenskit import batch, topn
import lenskit.crossfold as xf
from lenskit.algorithms import als, funksvd, item_knn, basic, Predictor, hpf

_log = logging.getLogger('exp.sweep')


def _run_algo(runid, train, test, algo, fields):
    start_time = time.perf_counter()
    model = algo.train(train)
    train_time = time.perf_counter()
    _log.info('trained model in %.2fs', train_time - start_time)
    if isinstance(algo, Predictor):
        preds = batch.predict(algo, test, model)
        preds['RunId'] = runid
    else:
        preds = None
    pred_time = time.perf_counter()
    _log.info('computed predictions in %.2fs', pred_time - train_time)
    recs = batch.recommend(algo, model, test.user.unique(), 100,
                           topn.UnratedCandidates(train), test)
    recs['RunId'] = runid
    rec_time = time.perf_counter()
    _log.info('computed recommendations in %.2fs', rec_time - pred_time)
    run = {'RunId': runid, 'Algorithm': algo.__class__.__name__, 'AlgoStr': str(algo),
           'TrainTime': train_time - start_time, 'PredictTime': pred_time - train_time,
           'RecTime': rec_time - pred_time}
    for f in fields:
        run[f] = getattr(algo, f, None)
    return run, preds, recs


def sweep(base, data, instances, fields):
    "Sweep over a set of instances using data."
    runid = 0

    for part, (train, test) in enumerate(xf.partition_users(data, 5, xf.SampleN(5))):
        runs = []
        preds = []
        recs = []

        for algo in instances:
            runid += 1
            _log.info('evaluating %s on partition %d', algo, part)
            r, ps, rs = _run_algo(runid, train, test, algo, fields)
            r['Partition'] = part
            runs.append(r)
            if ps is not None:
                preds.append(ps)
            if rs is not None:
                recs.append(rs)

        fpq.write(base + '-runs.parquet', pd.DataFrame(runs), append=part > 0, compression='snappy')
        fpq.write(base + '-preds.parquet', pd.concat(preds), append=part > 0, compression='snappy')
        fpq.write(base + '-recs.parquet', pd.concat(recs), append=part > 0, compression='snappy')


def sweep_als(data, base):
    "Sweep the ALS MF algorithm."
    sizes = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    regs = [0.01, 0.05, 0.1]
    instances = [basic.Bias(damping=5)]
    instances += (als.BiasedMF(sz, iterations=20, reg=reg)
                  for sz in sizes
                  for reg in regs)

    sweep(base, data, instances, ['features', 'regularization'])


def sweep_mf_all(data, base):
    "Sweep a suite of MF algorithms."
    sizes = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    instances = [basic.Bias(damping=5), basic.Popular()]
    instances += (als.BiasedMF(sz, iterations=20, reg=0.1)
                  for sz in sizes)
    instances += (als.ImplicitMF(sz, iterations=20, reg=0.1)
                  for sz in sizes)
    instances += (funksvd.FunkSVD(sz, iterations=125) for sz in sizes)
    instances += (hpf.HPF(sz) for sz in sizes)

    sweep(base, data, instances, ['features'])


def sweep_item_item(data, base):
    "Sweep the item-item k-NN algorithm."
    sizes = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    instances = [item_knn.ItemItem(n) for n in sizes]

    sweep(base, data, instances, ['max_neighbors'])

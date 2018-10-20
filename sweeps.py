import time
import logging

import pandas as pd
import fastparquet as fpq

from lenskit import batch, topn
import lenskit.crossfold as xf
from lenskit.algorithms import als, funksvd, item_knn, basic, Predictor, hpf

_log = logging.getLogger('exp.sweep')

def sweep(base, dsn, data, instances, fields):
    "Sweep over a set of instances using data."
    sweep = batch.MultiEval(base)

    sweep.add_datasets(xf.partition_users(data, 5, xf.SampleN(5)), name=dsn)
    sweep.add_algorithms(instances, attrs=fields)

    sweep.run()


def sweep_als(data, base, dsname):
    "Sweep the ALS MF algorithm."
    sizes = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    regs = [0.01, 0.05, 0.1]
    instances = [basic.Bias(damping=5)]
    instances += (als.BiasedMF(sz, iterations=20, reg=reg)
                  for sz in sizes
                  for reg in regs)

    sweep(base, dsname, data, instances, ['features', 'regularization'])


def sweep_mf_all(data, base, dsname):
    "Sweep a suite of MF algorithms."
    sizes = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    instances = [basic.Bias(damping=5), basic.Popular()]
    instances += (als.BiasedMF(sz, iterations=20, reg=0.1)
                  for sz in sizes)
    instances += (als.ImplicitMF(sz, iterations=20, reg=0.1)
                  for sz in sizes)
    instances += (funksvd.FunkSVD(sz, iterations=125) for sz in sizes)
    instances += (hpf.HPF(sz) for sz in sizes)

    sweep(base, dsname, data, instances, ['features'])


def sweep_item_item(data, base, dsname):
    "Sweep the item-item k-NN algorithm."
    sizes = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150]
    instances = [item_knn.ItemItem(n) for n in sizes]

    sweep(base, dsname, data, instances, ['max_neighbors'])

import datasets, algorithms

algo = algorithms.bpr
data = datasets.ml_small()

algo.fit(data)

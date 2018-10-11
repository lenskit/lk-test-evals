import datasets, algorithms

algo = algorithms.als
data = datasets.ml_small()

model = algo.train(data)

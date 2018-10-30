import datasets, algorithms

algo = algorithms.item_item
data = datasets.ml_small()

model = algo.train(data)

from lenskit.algorithms import item_knn, funksvd, basic

bias = basic.Bias(damping=5)

item_item = item_knn.ItemItem(20, min_sim=0)

funksvd = funksvd.FunkSVD(50, iterations=125, lrate=0.001)
from lenskit.algorithms import item_knn, user_knn, funksvd, basic, als

bias = basic.Bias(damping=5)

user_user = user_knn.UserUser(30, min_sim=1.0e-6)

item_item = item_knn.ItemItem(20, min_sim=1.0e-6, save_nbrs=5000)

funksvd = funksvd.FunkSVD(15, iterations=125, lrate=0.001)

als = als.BiasedMF(50, iterations=30, reg=0.1)

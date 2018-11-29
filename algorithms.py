from lenskit.algorithms import item_knn, user_knn, funksvd, basic, als as alsm

bias = basic.Bias(damping=5)

user_user = user_knn.UserUser(30, min_sim=1.0e-6)

item_item = item_knn.ItemItem(20, min_sim=1.0e-6, save_nbrs=5000)

funksvd = funksvd.FunkSVD(15, iterations=125, lrate=0.001)

als = alsm.BiasedMF(50, iterations=20, reg=0.1)
als_implicit = alsm.ImplicitMF(50, iterations=20, reg=0.1)

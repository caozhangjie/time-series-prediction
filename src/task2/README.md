K-means algorithm 

Kmeans.m: the main algorithm implementation.
Fscore.m: the implementation of F-score evaluation.
cPurity.m: the implementation of purity evaluation.
run_kmeans.m: running this file in Matlab and the results are saved in result.mat. You can set the k you want to run in the "test_k" variable. In the results, "all_label" is the cluster number of each data points cluster by K-means with respect to k. "all_center" is the mean of each cluster with respect to k. "purity", "f_score" and "iter_num" are the purity, F-score and number of iterations to coverage with respect to k.
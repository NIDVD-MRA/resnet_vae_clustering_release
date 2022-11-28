from sklearn.cluster import KMeans
import numpy as np
from clustering import em_k_means


def k_means(num_clusters, matrix):
    print('clustering with k-means')
    kmeans = KMeans(n_clusters=num_clusters, max_iter=5000000).fit(matrix)
    print('clustering labels: \n', kmeans.labels_)
    print('k-means finished')
    return kmeans.labels_





def my_AE_rotate_k_means(num_clusters, mrcarray, rotate_times, model):
    print('clustering with k-means++')
    k_means_mrcsAaaay, subCenter, center = em_k_means.AEcode_rotating_kmeans(mrcarray, num_clusters, rotate_times,
                                                                             model)
    labels = np.asarray(subCenter[:, 0].ravel(), dtype=int)
    labels = labels[0]
    print('clustering labels: \n', labels)
    print('k-means finished')
    return k_means_mrcsAaaay, labels


def dynAE_rotate_k_means(num_clusters, mrcarray, rotate_times, model):
    print('clustering with k-means++')
    k_means_mrcsAarry, subCenter, is_reliable_list, center = em_k_means.dynAEcode_rotating_kmeans(mrcarray,
                                                                                                  num_clusters,
                                                                                                  rotate_times, model)
    labels = np.asarray(subCenter[:, 0].ravel(), dtype=int)
    labels = labels[0]
    print('clustering labels: \n', labels)
    print('k-means finished')
    return k_means_mrcsAarry, labels, is_reliable_list

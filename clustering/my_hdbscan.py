# @Author : Written/modified by Yang Yan
# @Time   : 2022/6/21 下午2:17
# @E-mail : yanyang98@yeah.net
# @Function :
import hdbscan
def hdbscan_clustering(features,min_size=10):
    myhdbscan = hdbscan.HDBSCAN(min_cluster_size=10)
    clustering_labels_dbscan = myhdbscan.fit_predict(features)
    return clustering_labels_dbscan
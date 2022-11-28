import numpy as np
from random import random
from cryoemdata.process import mrcdata_process
from tqdm import tqdm
import Running_Paras
from sklearn.cluster import kmeans_plusplus, MiniBatchKMeans
from munkres import Munkres
from sklearn import metrics
import os


# import kmc2
def get_class_dis_averages(subCenter, k):
    dis_averages = np.zeros(8)
    labels = np.asarray(subCenter[:, 0].ravel(), dtype=int)
    labels = labels[0]
    dis = np.asarray(subCenter[:, 1].ravel(), dtype=float)
    dis = dis[0]
    for i in range(k):
        i_dis = dis[labels == i]
        dis_average = np.expand_dims(np.mean(i_dis, 0), axis=0)
        # save the averages
        dis_averages[i] = dis_average
    return dis_averages


def randomCenter(data, k):
    '''
    随机初始化聚类中心
    :param data: 训练数据
    :param k: 聚类中心的个数
    :return: 返回初始化的聚类中心
    '''
    n = np.shape(data)[1]  # 特征的个数
    cent = np.mat(np.zeros((k, n)))  # 初始化K个聚类中心
    for j in range(n):  # 初始化聚类中心每一维的坐标
        minJ = np.min(data[:, j])
        rangeJ = np.max(data[:, j]) - minJ
        cent[:, j] = minJ * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * rangeJ  # 在最大值和最小值之间初始化
    return cent


def class_stdVar(raw_images, reliable_list, labels,cluster_num=10):
    # clustering_num = Running_Paras.cluster_num
    class_number_arry = []
    # averages = raw_images[0]
    for i in range(cluster_num):
        # find children and generate averages
        # print(np.asarray(reliable_list)[labels == i])
        children = raw_images[labels == i]
        # children = children[labels == i]
        # label_children=reliable_list
        # print('class' + str(i) + ' ' + str(children.shape[0]))
        class_number_arry = class_number_arry + [children.shape[0]]
    stdVar = np.std(class_number_arry, ddof=1)
    # print('Standard deviation between classes ' + str(stdVar))
    return stdVar


def get_cent(points, k):
    '''
    kmeans++的初始化聚类中心的方法
    :param points: 样本
    :param k: 聚类中心的个数
    :return: 初始化后的聚类中心
    '''
    m, n = np.shape(points)
    center_index = []
    cluster_centers = np.mat(np.zeros((k, n)))

    # 1、随机选择一个样本点作为第一个聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0,] = np.copy(points[index])  # 复制函数，修改cluster_centers，不会影响points
    center_index.append(index)
    # 2、初始化一个距离序列
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            # 3、对每一个样本找到最近的聚类中心点
            d[j] = nearest(points[j,], cluster_centers[0:i, ])
            # 4、将所有的最短距离相加
            sum_all += d[j]
        # 5、取得sum_all之间的随机值
        sum_all *= random()
        # 6、获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j,])
            center_index.append(j)
            break
    return cluster_centers, center_index


def nearest(point, cluster_centers):
    '''
    计算point和cluster_centers之间的最小距离
    :param point: 当前的样本点
    :param cluster_centers: 当前已经初始化的聚类中心
    :return: 返回point与当前聚类中心的最短距离
    '''
    min_dist = np.inf
    m = np.shape(cluster_centers)[0]  # 当前已经初始化聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = np.linalg.norm(point - cluster_centers[i,])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist


def get_points_nearest_to_centers(points_data, centers, labels):
    k = centers.shape[0]
    points_nearest_to_centers = np.zeros(k)
    mindist = np.full(k, np.inf)
    for i in range(points_data.shape[0]):
        class_id = labels[i]
        dist = np.linalg.norm(points_data[i] - centers[class_id])
        if dist < mindist[class_id]:
            points_nearest_to_centers[class_id] = i
            mindist[class_id] = dist
    return points_nearest_to_centers


def circle_features_kmeans(mrcArray_circle_features, k, is_simulate, labels_true):
    '''
    kmeans算法求解聚类中心
    :param mrcflatArray: 训练数据
    :param k: 聚类中心的个数
    :param cent: 随机初始化的聚类中心
    :return: 返回训练完成的聚类中心和每个样本所属的类别
    '''
    stdVar1 = 0
    stdVar2 = 0
    acc = 0
    nmi = 0
    var = Running_Paras.var
    # Get the centroid of each cluster
    # cent, cent_index = get_cent(mrcArray_circle_features, k)
    cent, cent_index = kmeans_plusplus(mrcArray_circle_features, k)
    print(type(cent), cent.shape, cent_index)
    num_class = [0 for i in range(k)]

    number_of_samples, dimension_of_features = mrcArray_circle_features.shape  # m：样本的个数；n：特征的维度
    subCenter = np.mat(np.zeros((number_of_samples, 2)))  # 初始化每个样本所属的类别
    isreliable_list = [False] * number_of_samples
    # Set the flag of clustering
    change = True  # 判断是否需要重新计算聚类中心
    number_iterations = 0
    while change == True:
        num_changed_points = 0
        number_iterations = number_iterations + 1
        if number_iterations >= Running_Paras.max_iteration_times:
            change = False
        phbar = tqdm(range(number_of_samples))
        phbar.set_description("iteration" + str(number_iterations))
        for i in phbar:
            minDist = np.inf  # 设置样本与聚类中心的最小距离，初始值为正无穷
            minIndex = 0  # 所属的类别

            circle_features = mrcArray_circle_features[i]
            for j in range(k):
                dist = np.linalg.norm(circle_features - cent[j,])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            # 判断是否需要改变
            new_subCenter = subCenter.copy()
            if new_subCenter[i, 0] == minIndex:
                subCenter[i, 1] = minDist
            if new_subCenter[i, 0] != minIndex:  # 需要改变
                num_changed_points = num_changed_points + 1
                new_subCenter[i] = np.mat([minIndex, minDist])
                labels = np.asarray(subCenter[:, 0].ravel(), dtype=int)
                labels = labels[0]
                new_labels = np.asarray(new_subCenter[:, 0].ravel(), dtype=int)
                new_labels = new_labels[0]
                stdVar1 = class_stdVar(mrcArray_circle_features, [True] * number_of_samples, labels,cluster_num=k)
                stdVar2 = class_stdVar(mrcArray_circle_features, [True] * number_of_samples, new_labels,cluster_num=k)
                # dont care the std of class size
                # subCenter = new_subCenter

                # care the std of class size
                class_dist = np.asarray(subCenter[:, 1])
                class_dist = class_dist[labels == minIndex]
                mean_class_dist = np.mean(class_dist)
                if stdVar2 <= stdVar1:
                    subCenter = new_subCenter
                elif random() < (mean_class_dist / minDist) * 0.5:
                    subCenter = new_subCenter
        if num_changed_points < 5:

            print(str(np.asarray(num_class).std()) + str(np.asarray(num_class)))
            # analyse the performance of iteration
            labels_predict = np.asarray(subCenter[:, 0].ravel(), dtype=int)
            labels_predict = labels_predict[0]
            if is_simulate == True:
                # labels_true = [[i] * Running_Paras.clustering_size for i in range(Running_Paras.clustering_num)]
                # labels_true = np.asarray(labels_true).reshape(
                #     int(Running_Paras.clustering_size * Running_Paras.clustering_num))
                acc, nmi = mrcdata_process.performance_clustering(labels_true, labels_predict)
                print('Ieration ' + str(number_iterations) + ' Acc:' + str(acc) + ' NMI:' + str(nmi) + ' std: ' + str(
                    stdVar1))
            else:
                print(' std: ' + str(stdVar1))
            print('converged')

            dis = np.asarray(subCenter[:, 1].ravel(), dtype=float)
            dis = dis[0]
            for i in range(k):
                i_dis = dis[labels_predict == i]
                dis_average = np.expand_dims(np.mean(i_dis, 0), axis=0)
                # save the averages
                if i == 0:
                    dis_averages = dis_average
                    # print(dis_average.shape, dis_averages.shape)
                else:
                    # print(dis_average.shape, dis_averages.shape)
                    dis_averages = np.append(dis_averages, dis_average, axis=0)

            for i in range(number_of_samples):
                i_id = labels_predict[i]
                i_class_num = num_class[i_id]
                if i_class_num == 0:
                    i_class_num = 1
                if dis[i] < var * dis_averages[i_id] * number_of_samples / (k * i_class_num):
                    isreliable_list[i] = True
            return mrcArray_circle_features, labels_predict, isreliable_list, cent, acc, nmi, np.asarray(num_class)

        # 重新计算聚类中心
        for j in range(k):
            sum_all = np.zeros((1, dimension_of_features))
            r = 0  # 每个类别中样本的个数
            for i in range(number_of_samples):
                if subCenter[i, 0] == j:  # 计算第j个类别
                    sum_all += mrcArray_circle_features[i,]
                    r += 1
            num_class[j] = r
            for z in range(dimension_of_features):
                try:
                    cent[j, z] = sum_all[0, z] / r
                except:
                    print("ZeroDivisionError: division by zero")
        # calculate the std and details
        print(str(np.asarray(num_class).std()) + str(np.asarray(num_class)))
        # analyse the performance of iteration
        labels_predict = np.asarray(subCenter[:, 0].ravel(), dtype=int)
        labels_predict = labels_predict[0]
        if is_simulate == True:
            # labels_true = [[i] * Running_Paras.clustering_size for i in range(Running_Paras.clustering_num)]
            # labels_true = np.asarray(labels_true).reshape(
            #     int(Running_Paras.clustering_size * Running_Paras.clustering_num))
            acc, nmi = mrcdata_process.performance_clustering(labels_true, labels_predict)
            print('Ieration ' + str(number_iterations) + ' Acc:' + str(acc) + ' NMI:' + str(nmi) + ' std: ' + str(
                stdVar1))
        else:
            print(' std: ' + str(stdVar1))
    # # calculate average distance
    # labels = np.asarray(subCenter[:, 0].ravel(), dtype=int)
    # labels = labels[0]
    dis = np.asarray(subCenter[:, 1].ravel(), dtype=float)
    dis = dis[0]
    for i in range(k):
        i_dis = dis[labels_predict == i]
        dis_average = np.expand_dims(np.mean(i_dis, 0), axis=0)
        # save the averages
        if i == 0:
            dis_averages = dis_average
            # print(dis_average.shape, dis_averages.shape)
        else:
            # print(dis_average.shape, dis_averages.shape)
            dis_averages = np.append(dis_averages, dis_average, axis=0)
    for i in range(number_of_samples):
        i_id = labels_predict[i]
        if dis[i] < var * dis_averages[i_id] * number_of_samples / (k * num_class[i_id]):
            isreliable_list[i] = True
    return mrcArray_circle_features, labels_predict, isreliable_list, cent, acc, nmi, np.asarray(num_class)


def features_kmeans_using_sklearn(mrcArray_features, labels_true, k):
    '''
    kmeans算法求解聚类中心
    :param mrcflatArray: 训练数据
    :param k: 聚类中心的个数
    :param cent: 随机初始化的聚类中心
    :return: 返回训练完成的聚类中心和每个样本所属的类别
    '''
    num_class = []
    dis_to_center = []
    # dis_to_center_average = []
    acc = nmi = 0
    n = len(mrcArray_features)
    # reliable_ratio = Running_Paras.reliable_ratio
    # is_reliable = [False] * n
    # cluster_size = int(n / k)
    # clf = KMeans(n_clusters=k)
    clf = MiniBatchKMeans(n_clusters=k)
    clf.fit(mrcArray_features)
    centers = clf.cluster_centers_
    labels_predict = clf.labels_
    for i in range(n):
        dis_to_center.append(np.linalg.norm(mrcArray_features[i] - centers[labels_predict[i]]))

    if np.max(labels_predict) == np.max(labels_true):
        acc, nmi = performance_clustering(labels_true, labels_predict)
    else:
        print('warning: true labels not matching the predicted labels')
    for i in range(k):
        num_class.append(np.sum(labels_predict == i))
        # i_dis = np.array(dis_to_center)[labels_predict == i]
        # dis_average = np.expand_dims(np.mean(i_dis, 0), axis=0)
        # dis_to_center_average.append(dis_average)
    return labels_predict, centers, acc, nmi, np.asarray(num_class)


def performance_clustering(labels_real, labels_predict):
    label_same = best_map(labels_real, labels_predict)
    count = np.sum(labels_real[:] == label_same[:])
    acc = count.astype(float) / (labels_real.shape[0])
    nmi = metrics.normalized_mutual_info_score(labels_real, label_same)
    return acc, nmi


def best_map(L1, L2):
    # L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def clustering_res_vis(features,labels,save_path,epoch,labels_true=None):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    # X_pca = PCA(n_components=2).fit_transform(features)
    if labels.ndim==1:
        labels=[labels]
    if labels_true is not None:
        num_data=len(labels)

        for i,label in enumerate(labels):

            plt.figure(figsize=(10, 10))
            plt.subplot(num_data+1,2,2*i+1)
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, label="t-SNE-"+str(i),s=3)
            plt.legend()
            plt.subplot(num_data+1,2,2*i+2)
            # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label, label="PCA",s=3)
            plt.legend()
        plt.subplot(num_data+1,2,2*num_data+1)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_true, label="t-SNE_t",s=3)
        plt.legend()
        plt.subplot(num_data+1,2,2*num_data+2)
        # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_true, label="PCA_t",s=3)
        plt.legend()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path+'/digits_tsne-pca_spectrum_epoch'+str(epoch)+'.png', dpi=120)
    else:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, label="t-SNE",s=3)
        plt.legend()
        plt.subplot(122)
        # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, label="PCA",s=3)
        plt.legend()
        plt.savefig(save_path+'/digits_tsne-pca_epoch'+str(epoch)+'.png', dpi=120)
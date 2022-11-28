# @Auther : Written/modified by Yang Yan
# @Time   : 2021/10/17
# @E-mail : yanyang98@yeah.net
# @Function : Some tools for evaluating the clustering results.

import numpy as np
from sklearn import metrics
# import torch
# import math
# import os
# import torch.nn.functional as F
import tqdm
# import PIL.Image
# from sklearn import metrics
# from scipy.optimize import linear_sum_assignment
# from munkres import Munkres
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from torchvision.utils import save_image
from EMAN2 import *
# import multiprocessing
# from functools import partial
import mrcfile

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    # ind = linear_assignment(w.max() - w)
    ind = linear_sum_assignment(w.max() - w)
    # c=[w[i, j] for i, j in ind]
    # m=[w[i,j] for i,j in zip(ind[0],ind[1])]
    nmi = metrics.normalized_mutual_info_score(Y_pred, Y)
    return sum([w[i,j] for i,j in zip(ind[0],ind[1])])*1.0/Y_pred.size, nmi

def classify_mrcs(labels, k, path_for_data_classify, path_for_classified_data_save):
    # import gc
    count = [0 for _ in range(k)]
    toal=0
    for root, dirs, files in os.walk(path_for_data_classify):
        if len(files)>0:
            files_bar=tqdm.tqdm(files)
            files_bar.set_description('classifying mrcs')
            # for index, file in enumerate(files):
            for index, file in enumerate(files_bar):
                if os.path.splitext(file)[-1] == '.mrc' or os.path.splitext(file)[-1] == '.mrcs':
                    with mrcfile.open(root + file) as mrc:
                        mrcdata = mrc.data
                        mrcdata_len=mrcdata.shape[0]
                        for i in range(k):
                            data_for_one_class = mrcdata[labels[toal:toal+mrcdata_len] == i]
                            if data_for_one_class.size > 0:
                                save_path = os.path.join(path_for_classified_data_save,
                                                         'class_' + str(i))
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path+'/')
                                class_particles = mrcfile.new(
                                    save_path + '/class_'+str(i)+'part_' + str(
                                        count[i]) + '.mrcs',
                                    np.asarray(data_for_one_class), overwrite=True)
                                class_particles.close()
                                count[i]+=1
                        toal += mrcdata_len
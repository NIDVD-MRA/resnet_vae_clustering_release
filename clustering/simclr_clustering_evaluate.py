# @Auther : Written/modified by Yang Yan
# @Time   : 2021/10/17
# @E-mail : yanyang98@yeah.net
# @Function : Some tools for evaluating the clustering results.

import numpy as np
import torch
import math
import os
import torch.nn.functional as F
import tqdm
from modeltrain.utils import AverageMeter, confusion_matrix
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from munkres import Munkres
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from EMAN2 import *
import multiprocessing
from functools import partial



def aligin_one_img(im, ref):
    im = EMNumPy.numpy2em(im)
    # im.process_inplace("normalize.edgemean")
    # if im["nx"]!=nx or im["ny"]!=ny :
    # 	im=im.get_clip(Region(old_div(-(nx-im["nx"]),2),old_div(-(ny-im["ny"]),2),nx,ny))
    # im.write_image("result/seq.mrc",-1)
    # ima=im.align("translational",ref0,{"nozero":1,"maxshift":old_div(ref0["nx"],4.0)},"ccc",{})
    ima = im.align("rotate_translate_tree", ref)
    # ima.write_image("seq.mrc", -1)
    # print(fsp, ima["xform.align2d"], ima.cmp("ccc", ref))
    ima.process_inplace("normalize.toimage", {"to": ref, "ignore_zero": 1})
    return ima


def imgs_align(imgs, labels,
               k, pionts_nearest_to_centers, epoch, save_path, iteration_times=4):
    # plt.figure(figsize=(20, 20))
    # averages = raw_images[0]
    # rotate_angles = int(360 / rotate_times)
    phbar = tqdm.tqdm(range(k))
    phbar.set_description("rotating correction")
    aligined_imgs = imgs.copy()
    # num_imgs = imgs.shape[0]
    # for j in range(k):
    total_num = 0
    for j in phbar:
        # find children and generate averages
        children = imgs[labels == j]
        # children = children[np.asarray(reliable_list)[labels == j] == True]
        num_children = children.shape[0]
        # raw_imgs_children = raw_imgs[labels == j]
        ref = EMNumPy.numpy2em(imgs[int(pionts_nearest_to_centers[j])])
        for i in range(iteration_times):
            # print("Iter ", i)
            avgr = Averagers.get("mean", {"ignore0": True})
            item = [children[i] for i in range(num_children)]
            func = partial(aligin_one_img, ref=ref)
            pool = multiprocessing.Pool(10)
            aligined_img = pool.map(func, item)
            pool.close()
            pool.join()
            for num in range(len(aligined_img)):
                avgr.add_image(aligined_img[num])
                aligined_imgs[total_num + num] = EMNumPy.em2numpy(aligined_img[num])
            ref = avgr.finish()
        average = EMNumPy.em2numpy(ref)
        average = np.expand_dims(average, axis=0)
        # save the averages
        if j == 0:
            averages = average.copy()
        else:
            averages = np.append(averages, average, axis=0)
        total_num += num_children
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_image(torch.unsqueeze(torch.from_numpy(averages), 1),
               save_path + '/clustering_result' + str(epoch) + '.png')
    return averages


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


@torch.no_grad()
def kmeans_evaluate(val_loader, model, k):
    # top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    features_bank = []
    labels_bank = []
    for batch in val_loader:

        images = batch['mrcdata'].cuda(non_blocking=True)
        output = model(images)

        labels_true = batch['label']
        features_bank.append(output.cpu().numpy())
        labels_bank.append(labels_true.cpu().numpy())
    features_bank = np.concatenate(features_bank, axis=0)
    labels_bank = np.concatenate(labels_bank, axis=0)
    labels_predict, centers, acc, nmi, num_class = features_kmeans(features_bank, labels_bank, k)
    # if other_setings['get_averages']:
    #     # imgs_align(get_path_list(other_setings['classify_data_path']), labels_predict, k, get_points_nearest_to_centers(
    #     #     features_bank,
    #     #     centers, labels_predict), epoch, averages_save_path, other_setings['get_classified_mrcs'])
    #     other_setings['path_for_classified_data_save'] = os.path.join(save_path, 'classified_mrcs',
    #                                                                   'epoch_' + str(epoch))
    #     other_setings['path_averages_save'] = os.path.join(save_path, 'averages')
    #     # utils.utils.get_averages(other_setings)
    #     imgs_align(val_loader.dataset.images, labels_predict, k, get_points_nearest_to_centers(
    #         features_bank,
    #         centers, labels_predict), epoch, other_setings['path_averages_save'], other_setings['align_iteration_times'])
    return {'K-means ACC': acc, 'K-means NMI': nmi, 'num class': num_class,'features':features_bank,'predicted labels': labels_predict,'true labels': labels_bank}

def save_clustering_labels(labels,save_path,epoch):
    labels_path=os.path.join(save_path,'clustering_labels')
    if not os.path.exists(labels_path):
        os.makedirs(labels_path + '/')
    np.save(labels_path + '/predict_labels_epoch'+str(epoch)+'.npy',labels)
    # filename = open(labels_path + '/predict_labels_epoch'+str(epoch)+'.txt', 'w')
    # filename.write(str(labels))
    # filename.close()

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


def performance_clustering(labels_real, labels_predict):
    label_same = best_map(labels_real, labels_predict)
    count = np.sum(labels_real[:] == label_same[:])
    acc = count.astype(float) / (labels_real.shape[0])
    nmi = metrics.normalized_mutual_info_score(labels_real, label_same)
    return acc, nmi


def features_kmeans(mrcArray_features, labels_true, k):
    '''
    kmeans算法求解聚类中心
    :param mrcflatArray: 训练数据
    :param k: 聚类中心的个数
    :param cent: 随机初始化的聚类中心
    :return: 返回训练完成的聚类中心和每个样本所属的类别
    '''
    num_class = []
    dis_to_center = []
    dis_to_center_average = []
    acc = nmi = 0
    n = len(mrcArray_features)
    # reliable_ratio = Running_Paras.reliable_ratio
    # is_reliable = [False] * n
    cluster_size = int(n / k)
    clf = KMeans(n_clusters=k)
    clf.fit(mrcArray_features)
    centers = clf.cluster_centers_
    labels_predict = clf.labels_
    for i in range(n):
        dis_to_center.append(np.linalg.norm(mrcArray_features[i] - centers[labels_predict[i]]))

    if np.max(labels_predict) == np.max(labels_true):
        acc, nmi = performance_clustering(labels_true, labels_predict)
    for i in range(k):
        num_class.append(np.sum(labels_predict == i))
        # i_dis = np.array(dis_to_center)[labels_predict == i]
        # dis_average = np.expand_dims(np.mean(i_dis, 0), axis=0)
        # dis_to_center_average.append(dis_average)
    return labels_predict, centers, acc, nmi, np.asarray(num_class)


@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)
        output = model(images)
        output = memory_bank.weighted_knn(output)
        # output = memory_bank.knn(output)
        acc1 = 100 * torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), images.size(0))
    return top1.avg








@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, class_names=None,
                       compute_purity=True, compute_confusion_matrix=True,
                       confusion_matrix_file=None):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(),
                         class_names, confusion_matrix_file)

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

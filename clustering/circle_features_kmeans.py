import os
import sys
import time
import warnings
import numpy as np
from cryoemdata.process import mrcdata_process
import torch
from torchvision.utils import save_image
from easydict import EasyDict
import Running_Paras
from clustering import em_k_means
import yaml


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def to_img(x):
    x = x.view(x.size(0), 1, Running_Paras.resized_mrc_width, Running_Paras.resized_mrc_height)
    return x


# 运行设置
# sys.stdout = Logger('result/2021_03_02_19_39_01/a.log', sys.stdout)
# sys.stderr = Logger('result/2021_03_02_19_39_01/a.log_file', sys.stderr)

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == "__main__":
    # Set the root dir of this experiment
    '''get config'''
    cfg = EasyDict()
    with open('../settings.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    for k, v in config.items():
        cfg[k] = v

    # Set the root dir of this experiment
    time_of_run = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print(time_of_run)
    run_root_dir = cfg['path_result_dir']
    # old_model_path=cfg['old_model_path']
    path_data = cfg['path_data']
    cluster_num = cfg['cluster_num']
    epoches_round2 = cfg['epoches_round2']

    time_of_run = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    # print(time_of_run)
    # run_root_dir = './result/' + time_of_run + '/'

    if not os.path.exists(run_root_dir):
        os.makedirs(run_root_dir)

    # For calculating the time cost
    time_start = time.time()

    # Load and buid the dataset and labelsl;
    raw_mrcArrays, labels_true = mrcdata_process.loading_mrcs(path_data)
    masks = mrcdata_process.get_masks(Running_Paras.resized_mrc_width, Running_Paras.circle_features_num, Running_Paras.circle_feature_length)
    # filtered_raw_mrcArrays = filter.gauss_low_pass_filter(raw_mrcArrays.copy(), 10)
    # mask = mrcdata_preprocess.get_crop_mask(raw_mrcArrays.shape[1], Running_Paras.crop_ratio)
    # cropped_mrcArrays = mrcdata_preprocess.multi_process_crop(raw_mrcArrays.real, mask)
    # cropped_mrcArrays = mrcdata_preprocess.multi_process_crop(filtered_raw_mrcArrays.real, mask)
    mrcArrays_circle_features = mrcdata_process.get_mrcs_circle_features(raw_mrcArrays,
                                                                         masks)
    test_times = 0
    acc_best = 0
    average_acc=0
    average_nmi=0
    time_end_phase1 = time.time()
    while test_times < epoches_round2:
        test_times = test_times + 1

        k_means_mrcsArrays, clustering_labels, _,cent, acc,nmi, class_number_arry = em_k_means.circle_features_kmeans(
            mrcArrays_circle_features,
            cluster_num,
            True,labels_true
        )
        average_acc+=acc
        average_nmi+=nmi
        if acc > acc_best:
            acc_best = acc
            if not os.path.exists(run_root_dir):
                os.makedirs(run_root_dir)
            np.savetxt(run_root_dir + 'best_labels.txt', clustering_labels)
            with open(run_root_dir + 'best_record.txt', 'w') as best_record:
                best_record.write('test time:' + str(test_times) + '\n')
                best_record.write(str(acc_best) + '\n')
                best_record.write(str(class_number_arry) + '\n')

        # average_imgs_filtered = mrcdata_process.imgs_align(raw_mrcArrays,
        #                                                    [True] * raw_mrcArrays.shape[0],
        #                                                    labels=clustering_labels,
        #                                                    em_k_means.get_points_nearest_to_centers(
        #                                                        mrcArrays_circle_features,
        #                                                        cent,
        #                                                        clustering_labels))
        #
        # if not os.path.exists(run_root_dir + 'averages/'):
        #     os.makedirs(run_root_dir + 'averages/')
        # save_image(torch.unsqueeze(torch.from_numpy(average_imgs_filtered), 1),
        #            run_root_dir + 'averages/filtered_clustering_result_' + str(test_times) + '.png')

        average_imgs_all = mrcdata_process.imgs_align(imgs=raw_mrcArrays, k=cluster_num,
                                                      labels=clustering_labels,
                                                      pionts_nearest_to_centers=em_k_means.get_points_nearest_to_centers(
                                                          mrcArrays_circle_features,
                                                          cent, clustering_labels),path_result_dir=run_root_dir)
        # if not os.path.exists(run_root_dir + 'averages/'):
        #     os.makedirs(run_root_dir + 'averages/')
        # save_image(torch.unsqueeze(torch.from_numpy(average_imgs_all), 1),
        #            run_root_dir + 'averages/clustering_result_' + str(test_times) + '.png')
    time_end_phase2 = time.time()
    time_cost_phase2 = time_end_phase2 - time_end_phase1

    if not os.path.exists(run_root_dir):
        os.makedirs(run_root_dir)
    # np.savetxt(run_root_dir + 'best_labels.txt', clustering_labels)
    with open(run_root_dir + 'average_record.txt', 'w') as best_record:
        best_record.write('test time:' + str(time_cost_phase2/epoches_round2) + '\n')
        best_record.write('average acc:'+str(average_acc/epoches_round2) + '\n')
        best_record.write('average nmi:'+str(average_nmi/epoches_round2) + '\n')
# @Author : Written/modified by Yang Yan
# @Time   : 2022/1/17 下午6:41
# @E-mail : yanyang98@yeah.net
# @Function :
import os
import time
import numpy as np
import torch
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from cryoemdata import Loading_Dataset
from clustering import em_k_means
from cryoemdata.process import mrcdata_process
from models import AE_Models
from modeltrain import model_train_VAE


def test_with_trained_model(is_simulated_data, data_path, output_path, model_path, test_times, k,
                            circle_features_num=31):
    # Set the root dir of this experiment
    time_of_run = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print(time_of_run)
    output_dir = os.path.join(output_path, time_of_run) + '/'
    tb_writer = SummaryWriter(log_dir=output_dir + "tensorboard/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir + 'settings.txt', 'w') as settings:
        settings.write('dataset path:' + data_path + '\n')
        settings.write('clustering times:' + str(test_times) + '\n')
        settings.write('model path:' + str(model_path) + '\n')
    # Build the model and put it on cuda
    model = AE_Models.VAE_RESNET18(z_dim=128)
    if torch.cuda.is_available():
        model.cuda()
        print("Is model on gpu: {}".format(next(model.parameters()).is_cuda))
    print(model)
    # Initialate the paras of model
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(layer.weight, val=1.0)
            torch.nn.init.constant_(layer.bias, val=0.0)

    # Set the hyper-paramaters
    batch_size = 40
    nw = min(os.cpu_count(), 60)  # number of workers
    print('Using {} dataloader workers'.format(nw))

    # Load and buid the dataset and labels
    raw_mrcArrays, labels_true = mrcdata_process.loading_mrcs(data_path)
    isreliable_list = [False] * raw_mrcArrays.shape[0]
    test_dataset = Loading_Dataset.Dataset_phase1(images=raw_mrcArrays,
                                                  images_class=np.zeros((raw_mrcArrays.shape[0],), dtype=int),
                                                  isreliable_list=isreliable_list,
                                                  transform=transforms.Compose([
                                                       transforms.ToTensor(),
                                                       transforms.RandomRotation(degrees=(-180, 180)),
                                                       transforms.Normalize(raw_mrcArrays.mean(),
                                                                            raw_mrcArrays.std())
                                                   ]))

    testdata_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model, optimizer = load_model(
        model_path,
        model, optimizer)
    generated_imgs = model_train_VAE.get_generated(model, testdata_loader)
    input_data_of_features_extraction = np.squeeze(generated_imgs)

    mrcArrays_circle_features = mrcdata_process.get_mrcs_circle_features(input_data_of_features_extraction,
                                                                         circle_features_num, 2)
    acc_best = 0
    nmi_best = 0
    acc_sum = 0
    nmi_sum = 0
    clustering_times = 0

    # For calculating the time cost
    time_start = time.time()

    for times in range(test_times):
        k_means_mrcsArrays, clustering_labels, isreliable_list, cent, acc, nmi, class_number_arry = em_k_means.circle_features_kmeans(
            mrcArrays_circle_features,
            k, is_simulated_data, labels_true)
        labels_path = os.path.join(output_dir, 'clustering_labels')
        if not os.path.exists(labels_path):
            os.makedirs(labels_path + '/')
        np.save(labels_path + '/predict_labels_epoch' + str(times) + '.npy', clustering_labels)
        if is_simulated_data:
            acc_best, nmi_best, acc_sum, nmi_sum, clustering_times = model_train_VAE.save_acc_data(acc, acc_best, nmi,
                                                                                                   nmi_best,
                                                                                                   tb_writer, acc_sum,
                                                                                                   nmi_sum,
                                                                                                   clustering_times, times,
                                                                                                   output_dir)
        average_imgs_all, algined_raw_imgs, labels_aligned = mrcdata_process.imgs_align(
            raw_mrcArrays.copy(), clustering_labels, k,
            em_k_means.get_points_nearest_to_centers(
                mrcArrays_circle_features,
                cent,
                clustering_labels))
        average_generated_imgs, _, _ = mrcdata_process.imgs_align(
            input_data_of_features_extraction.copy(), clustering_labels,
            k,
            em_k_means.get_points_nearest_to_centers(
                mrcArrays_circle_features,
                cent,
                clustering_labels))
        model_train_VAE.save_averages(average_imgs_all, average_generated_imgs, output_dir, times)
    time_end = time.time()
    time_cost_all = time_end - time_start
    with open(output_dir + 'time_cost.txt', 'w') as time_cost:
        time_cost.write('time cost:' + str(time_cost_all) + 's\n')
    print("The expriment cost " + str(time_cost_all) + "s")


def load_model(model_path, model, optimizer):
    if os.path.exists(model_path):
        print('Restart from ' + model_path)
        checkpoint = torch.load(model_path, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        # model = torch.nn.DataParallel(model)
        # model.to(device)
        return model, optimizer
    else:
        print('No checkpoint file found')


if __name__ == "__main__":
    data_path_emd6840_01 = "/home/yanyang/data/em_dataset/simulated_particles/with_translation/std3/emd_6840/snr_0.1/"
    data_path_emd6840_01_model = "/home/yanyang/data/project/resnet_vae_clustering/result/formal_emd6840_01_translate3_2022_1_18_test1/trained_model/epoch90_model.pth.tar"
    test_with_trained_model(True, data_path_emd6840_01, './trained_model_test/', data_path_emd6840_01_model, 10, 10)

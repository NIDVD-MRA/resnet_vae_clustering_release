import os
import sys
import time
import warnings
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from cryoemdata import Loading_Dataset
import Running_Paras
from clustering import em_k_means
from cryoemdata.process import mrcdata_process
from models import AE_Models
import mrcfile
# from torchsummary import summary


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
sys.stdout = Logger('../result/a.log', sys.stdout)
sys.stderr = Logger('../result/a.log_file', sys.stderr)

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def get_features(net, mrcs_data, epoch):
    net.eval()
    feature_bank = []
    with torch.no_grad():
        for data, _, _ in tqdm(mrcs_data, desc='Feature extracting'):
            feature, _, _ = net.forward(data.cuda())
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0)
        feature_bank = feature_bank.cpu().numpy()
        # feature_len = int(feature_bank.shape[0] / 1000)
        # for i in range(feature_len):
        #     plt.subplot(int(feature_len ** 0.5) + 1, int(feature_len ** 0.5) + 1, i + 1)
        #     plt.imshow(feature_bank[i][0], cmap='gray')
        #     plt.xticks([])  # 去掉横坐标值
        #     plt.yticks([])  # 去掉纵坐标值
        # plt.savefig('samples' + str(epoch) + '.png', cmap='gray', dpi=500)
    return feature_bank


if __name__ == "__main__":
    # Set the root dir of this experiment
    time_of_run = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print(time_of_run)
    run_root_dir = './result/' + time_of_run + '/'
    tb_writer = SummaryWriter(log_dir=run_root_dir + "tensorboard/")
    if not os.path.exists(run_root_dir):
        os.makedirs(run_root_dir)

    with open(run_root_dir + 'settings.txt', 'w') as settings:
        settings.write('dataset path:' + Running_Paras.path_noisy_simulated_data + '\n')
        settings.write('var:' + str(Running_Paras.var) + '\n')
        settings.write('clustering_interval:' + str(Running_Paras.clustering_interval) + '\n')
        settings.write('epoches_round1:' + str(Running_Paras.epoches_round1) + '\n')
        settings.write('epoches_round2:' + str(Running_Paras.epoches_round2) + '\n')
    # For calculating the time cost
    time_start_phase1 = time.time()
    model = AE_Models.VAE_RESNET18(z_dim=128)
    if torch.cuda.is_available():
        model.cuda()
        model = nn.DataParallel(model)
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
    batch_size = 30
    lr = 1e-3
    weight_decay = 1e-5
    epoches_round1 = Running_Paras.epoches_round1
    epoches_round2 = Running_Paras.epoches_round2
    clustering_times = 0
    nw = min(os.cpu_count(), 60)  # number of workers
    print('Using {} dataloader workers'.format(nw))

    # Load and buid the dataset and labels
    raw_mrcArrays, labels_true = mrcdata_process.loading_mrcs(Running_Paras.path_noisy_simulated_data)
    # filtered_raw_mrcArrays = filter.gauss_low_pass_filter(raw_mrcArrays.copy(), 25)
    # cropped_mrcArrays = mrcdata_process.multi_process_crop(filtered_raw_mrcArrays)
    raw_mrcArrays = mrcdata_process.multi_process_crop(raw_mrcArrays)
    # input_data_of_features_extraction = cropped_mrcArrays
    # cropped_mrcArrays = raw_mrcArrays
    input_data_of_features_extraction = raw_mrcArrays
    isreliable_list = [False] * raw_mrcArrays.shape[0]
    corrected_mrcs = input_data_of_features_extraction

    train_data_set = Loading_Dataset.Dataset_phase1(images=input_data_of_features_extraction,
                                                    images_class=np.zeros((raw_mrcArrays.shape[0],), dtype=int),
                                                    isreliable_list=isreliable_list,
                                                    transform=transforms.Compose([
                                                         transforms.ToTensor(),
                                                         # transforms.Normalize(mrcArrays_mean, mrcArrays_std)
                                                     ]))
    train_data = torch.utils.data.DataLoader(train_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             )
    data_for_clustering = torch.utils.data.DataLoader(train_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      )

    stdVar = 0
    acc_best = 0
    acc_sum = 0
    nmi_sum = 0
    optimizier = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epoches_round1):
        # 模型开始训练
        model.train(True)
        for img, label, is_reliable in train_data:
            # img = torch.flatten(img, start_dim=1)
            img = Variable(img).cuda()
            # forward
            unreliable_img = img[is_reliable == False]
            num_all_img = img.shape[0]
            num_unreliable_img = unreliable_img.shape[0]
            loss_unreliable = 0
            # For VAE
            gen_unreliable_img, mu_unreliable_img, logvar_unreliable_img = model.forward(unreliable_img)
            loss_unreliable = AE_Models.loss_VAE(gen_unreliable_img, unreliable_img, mu_unreliable_img,
                                                 logvar_unreliable_img, 0.5) / num_unreliable_img
            tb_writer.add_scalar("round1 loss:", loss_unreliable, epoch)
            loss = loss_unreliable
            tb_writer.add_scalar("round1 loss:", loss, epoch)
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        # 每五个循环保存编码图像，从而检视AE效果
        if epoch % 5 == 0:
            # Build dataset
            if not os.path.exists(run_root_dir + 'AE_iterations/'):
                os.makedirs(run_root_dir + 'AE_iterations/')
                # Show the unre_imgs of last batch and its generated imgs
            if num_unreliable_img != 0:
                unreliable_img_gen = torch.cat([unreliable_img, gen_unreliable_img], dim=0)
                save_image(to_img(unreliable_img_gen),
                           run_root_dir + 'AE_iterations/' + 'round1_epoch_{}_unreimg_gen.png'.format(
                               epoch + 1))
            # Show the re_imgs of last batch and its generated imgs
        # Get the loss of each epoch
        record_loss_epoch = str("epoch: {}, loss is {}, lr is {} \n".format((epoch + 1), loss.data.float(),
                                                                            optimizier.param_groups[0]['lr']))
        # Show the loss of epoch
        print(record_loss_epoch)

    time_end_phase1 = time.time()
    time_cost_phase1 = time_end_phase1 - time_start_phase1
    with open(run_root_dir + 'time_cost.txt', 'w') as time_cost:
        time_cost.write('phase 1:' + str(time_cost_phase1) + 's\n')

    if Running_Paras.training_in_phase_2:
        generated_imgs = get_features(model, data_for_clustering, epoch)
        input_data_of_features_extraction = np.squeeze(generated_imgs)
    for epoch in range(epoches_round2):
        if Running_Paras.training_in_phase_2 | (epoch % Running_Paras.clustering_interval == 0):
            if bool(1-Running_Paras.training_in_phase_2):
                generated_imgs = get_features(model, data_for_clustering, epoch)
                input_data_of_features_extraction = np.squeeze(generated_imgs)

            mrcArrays_circle_features = mrcdata_process.get_mrcs_circle_features(input_data_of_features_extraction,
                                                                                 Running_Paras.circle_features_num,
                                                                                 Running_Paras.circle_feature_length)
            k_means_mrcsArrays, clustering_labels, isreliable_list, cent, acc, nmi, class_number_arry = em_k_means.circle_features_kmeans(
                mrcArrays_circle_features,
                Running_Paras.cluster_num, True)
            tb_writer.add_scalar("accuracy:", acc, epoch)
            acc_sum = acc_sum + acc
            nmi_sum = nmi + nmi_sum
            clustering_times = clustering_times + 1
            tb_writer.add_scalar("mean accuracy:", acc_sum / clustering_times,
                                 clustering_times)
            tb_writer.add_scalar("mean NMI:", nmi_sum / clustering_times,
                                 clustering_times)
            if not os.path.exists(run_root_dir + 'best/'):
                os.makedirs(run_root_dir + 'best/')
            if acc > acc_best:
                acc_best = acc
                # if not os.path.exists(run_root_dir + 'best/'):
                #     os.makedirs(run_root_dir + 'best/')
                np.savetxt(run_root_dir + 'best/' + 'best_labels.txt', clustering_labels)
                with open(run_root_dir + 'best/' + 'best_record.txt', 'w') as best_record:
                    best_record.write('epoch:' + str(epoch) + '\n')
                    best_record.write(str(acc_best) + '\n')
                    best_record.write('nmi:' + str(nmi) + '\n')
                    best_record.write(str(class_number_arry) + '\n')
            with open(run_root_dir + 'best/' + 'average_acc.txt', 'w') as average_acc:
                average_acc.write('average accuracy' + str(acc_sum / clustering_times))
                average_acc.write('\naverage NMI' + str(nmi_sum / clustering_times))
            if not os.path.exists(run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/'):
                os.makedirs(run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/')
            if Running_Paras.is_save_clustered_single_particles:
                for i in range(Running_Paras.cluster_num):
                    class_single_particles = raw_mrcArrays[clustering_labels == i]
                    generated_class_single_particles = generated_imgs[clustering_labels == i]

                    class_particles = mrcfile.new(
                        run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/' + 'class_' + str(
                            i) + "_" + str(class_number_arry[i]) + '.mrcs',
                        generated_class_single_particles, overwrite=True)
                    class_particles.close()
            average_generated_imgs, average_imgs_all, algined_raw_imgs = mrcdata_process.get_averages_with_rotate_correction_with_generated_imgs(
                input_data_of_features_extraction, raw_mrcArrays,
                [True] * raw_mrcArrays.shape[0],
                clustering_labels,
                em_k_means.get_points_nearest_to_centers(
                    mrcArrays_circle_features,
                    cent, clustering_labels), True)
            if not os.path.exists(run_root_dir + 'averages/'):
                os.makedirs(run_root_dir + 'averages/')
            save_image(torch.unsqueeze(torch.from_numpy(average_imgs_all), 1),
                       run_root_dir + 'averages/clustering_result_' + str(epoch) + '.png')
            save_image(torch.unsqueeze(torch.from_numpy(average_generated_imgs), 1),
                       run_root_dir + 'averages/generated_clustering_result_' + str(epoch) + '.png')

            train_data_set = Loading_Dataset.Dataset_phase1(images=algined_raw_imgs,
                                                            images_class=clustering_labels,
                                                            isreliable_list=isreliable_list,
                                                            transform=transforms.Compose([
                                                                 transforms.ToTensor(),
                                                                 # transforms.Normalize(mrcArrays_mean, mrcArrays_std)
                                                             ]))
            train_data = torch.utils.data.DataLoader(train_data_set,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     num_workers=nw,
                                                     )

        # 模型开始训练
        if bool(1-Running_Paras.training_in_phase_2):
            model.train(True)
            for img, label, is_reliable in train_data:
                # img = torch.flatten(img, start_dim=1)
                img = Variable(img).cuda()
                # forward
                reliable_img = img[is_reliable == True]
                reliable_labels = label[is_reliable == True]
                unreliable_img = img[is_reliable == False]
                num_all_img = img.shape[0]
                num_reliable_img = reliable_img.shape[0]
                num_unreliable_img = unreliable_img.shape[0]
                loss_reliable = 0
                loss_unreliable = 0
                if num_reliable_img > 6:
                    average_img = torch.from_numpy(average_imgs_all[reliable_labels])
                    # average_img = torch.flatten(average_img, start_dim=1)
                    average_img = torch.unsqueeze(average_img, 1).cuda()
                    # For VAE
                    gen_reliable_img, mu_reliable_img, logvar_reliable_img = model.forward(reliable_img)
                    # loss_reliable = AE_Models.loss_func_VAE_RESNET(gen_reliable_img, average_img, mu_reliable_img,
                    #                                    logvar_reliable_img) / num_reliable_img
                    loss_reliable = (AE_Models.loss_VAE(gen_reliable_img, average_img, mu_reliable_img,
                                                        logvar_reliable_img,
                                                        0.5) / num_reliable_img + AE_Models.loss_VAE(gen_reliable_img,
                                                                                                     reliable_img,
                                                                                                     mu_reliable_img,
                                                                                                     logvar_reliable_img,
                                                                                                     0.5) / num_unreliable_img) * 0.5
                    tb_writer.add_scalar("loss for reliable samples:", loss_reliable, epoch)
                    # For AE
                    # gen_reliable_img = model(reliable_img)
                    # loss_reliable = AE_Models.loss_MSE(gen_reliable_img, reliable_img) / num_reliable_img
                if num_unreliable_img != 0:
                    # For VAE
                    gen_unreliable_img, mu_unreliable_img, logvar_unreliable_img = model.forward(unreliable_img)
                    # loss_unreliable = AE_Models.loss_func_VAE_RESNET(gen_unreliable_img, unreliable_img, mu_unreliable_img,
                    #                                      logvar_unreliable_img) / num_unreliable_img
                    loss_unreliable = AE_Models.loss_VAE(gen_unreliable_img, unreliable_img, mu_unreliable_img,
                                                         logvar_unreliable_img, 0.5) / num_unreliable_img
                    tb_writer.add_scalar("loss for unreliable samples:", loss_unreliable, epoch)
                    # For AE
                    # gen_unreliable_img = model(unreliable_img)
                    # loss_unreliable = AE_Models.loss_MSE(gen_unreliable_img, unreliable_img) / num_unreliable_img
                # Real loss consists two part loss
                loss = (num_reliable_img / num_all_img) * loss_reliable + (
                        num_unreliable_img / num_all_img) * loss_unreliable
                tb_writer.add_scalar("round2 loss:", loss, epoch)
                optimizier.zero_grad()
                loss.backward()
                optimizier.step()

            # Get the loss of each epoch
            record_loss_epoch = str("epoch: {}, loss is {}, lr is {} \n".format((epoch + 1), loss.data.float(),
                                                                                optimizier.param_groups[0]['lr']))
            # Show the loss of epoch
            print(record_loss_epoch)

        # 每五个循环保存编码图像，从而检视AE效果
        if (epoch % 5 == 0)&(bool(1-Running_Paras.training_in_phase_2)):
            # Build dataset
            if not os.path.exists(run_root_dir + 'AE_iterations/'):
                os.makedirs(run_root_dir + 'AE_iterations/')
                # Show the unre_imgs of last batch and its generated imgs
            if num_unreliable_img != 0:
                unreliable_img_gen = torch.cat([unreliable_img, gen_unreliable_img], dim=0)
                save_image(to_img(unreliable_img_gen),
                           run_root_dir + 'AE_iterations/' + 'round2_epoch_{}_unreimg_gen.png'.format(
                               epoch + 1))
            # Show the re_imgs of last batch and its generated imgs
            if num_reliable_img > 6:
                reliable_img_np = torch.squeeze(reliable_img).cpu().numpy()
                gen_reliable_img_np = torch.squeeze(gen_reliable_img).cpu().detach().numpy()
                reliable_img_gen_average = torch.cat([reliable_img, gen_reliable_img, average_img], dim=0)
                save_image(to_img(reliable_img_gen_average),
                           run_root_dir + 'AE_iterations/' + 'round2_epoch_{}_reimg_gen_avg.png'.format(
                               epoch + 1))
                # Show the imgs of last batch and its generated imgs
                img_gen = torch.cat([img, gen_reliable_img, gen_unreliable_img], dim=0)
                save_image(to_img(img_gen),
                           run_root_dir + 'AE_iterations/' + 'round2_epoch_{}_img_gen.png'.format(
                               epoch + 1))
            else:
                # Show the imgs of last batch and its generated imgs
                img_gen = torch.cat([img, gen_unreliable_img], dim=0)
                save_image(to_img(img_gen),
                           run_root_dir + 'AE_iterations/' + 'round2_epoch_{}_img_gen.png'.format(
                               epoch + 1))

    # For calculating the time cost

    time_end_phase2 = time.time()
    time_cost_phase2 = time_end_phase2 - time_end_phase1
    time_cost_all = time_end_phase2 - time_start_phase1
    with open(run_root_dir + 'time_cost.txt', 'w') as time_cost:
        time_cost.write('phase 1:' + str(time_cost_phase1) + 's\n')
        time_cost.write('phase 2:' + str(time_cost_phase2) + 's\n')
        time_cost.write('all:' + str(time_cost_all) + 's\n')
    print("The expriment cost " + str(time_cost_all) + "s")

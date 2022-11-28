# @Auther : Written/modified by Yang Yan
# @Time   : 2021/12/22 上午10:23
# @E-mail : yanyang98@yeah.net
# @Function :
from torch.autograd import Variable
import torch
import os
from torchvision.utils import save_image
import Running_Paras
from models import AE_Models, life_long_learning_models
from tqdm import tqdm
import numpy as np
from clustering import em_k_means
import mrcfile
from cryoemdata import Loading_Dataset
from cryoemdata.process import evaluate_utils, mrcdata_process
import time
import copy
import get_config

'''convert a vector to matrix, now it is useless'''


def to_img(x):
    x = x.view(x.size(0), 1, Running_Paras.resized_mrc_width, Running_Paras.resized_mrc_height)
    return x


'''turn mrcs data to features'''


def get_generated(net, mrcs_data):
    net.eval()
    feature_bank = []
    with torch.no_grad():
        for data, _, _, _ in tqdm(mrcs_data, desc='Feature extracting'):
            feature, _, _ = net.forward(data.cuda())
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0)
        feature_bank = feature_bank.cpu().numpy()
    return feature_bank

'''training for round 1'''
def train_model_round1(start_epoch, epoches, model, train_data, run_root_dir, tb_writer, optimizer, old_dataloader=None,
                       using_MAS=False):
    if using_MAS:
        if old_dataloader is None:
            print('cannot find old dataset, MAS init failed.')
        else:
            mas = life_long_learning_models.MAS(copy.deepcopy(model), old_dataloader, torch.device('cuda'))
    else:
        mas = None
    for epoch in range(start_epoch, epoches):
        # 模型开始训练

        model.train(True)
        flag_for_save_images = True
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(len(train_data),
                                 [losses],
                                 prefix="Epoch: [{}]".format(epoch))
        for i, batch in enumerate(train_data):
            # img = torch.flatten(img, start_dim=1)
            img = batch['mrcdata']
            target = batch['target']
            is_reliable = batch['is_reliable']
            img = Variable(img).cuda()
            target = Variable(target).cuda()
            # forward
            unreliable_img = img[is_reliable == False]
            # num_all_img = img.shape[0]

            # For VAE
            gen_unreliable_img, mu_unreliable_img, logvar_unreliable_img = model.forward(unreliable_img)
            num_unreliable_img = unreliable_img.shape[0]
            # loss_unreliable = AE_Models.loss_VAE(gen_unreliable_img, target, mu_unreliable_img,
            #                                      logvar_unreliable_img, 0.5) / num_unreliable_img
            loss_unreliable = model.criterion_p1(gen_unreliable_img, target, mu_unreliable_img,
                                                 logvar_unreliable_img, 0.5) / num_unreliable_img
            loss = loss_unreliable
            if mas is not None:
                loss_mas = mas.penalty(model)
                loss += loss_mas
                tb_writer.add_scalar("round1 mas loss:", loss_mas, epoch)
            # tb_writer.add_scalar("round1 loss:", loss_unreliable, epoch)
            losses.update(loss.item())
            if i % 25 == 0:
                progress.display(i)

            tb_writer.add_scalar("round1 loss:", loss, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if flag_for_save_images:
                flag_for_save_images = False
                unreliable_img_gen = torch.cat([unreliable_img, target, gen_unreliable_img], dim=0)
        # 每五个循环保存编码图像，从而检视AE效果
        if epoch % 5 == 0:
            # Build dataset
            if not os.path.exists(run_root_dir + 'AE_iterations/'):
                os.makedirs(run_root_dir + 'AE_iterations/')
                # Show the unre_imgs of last batch and its generated imgs
            if num_unreliable_img != 0:
                save_image(to_img(unreliable_img_gen),
                           run_root_dir + 'AE_iterations/' + 'round1_epoch_{}_unreimg_gen.png'.format(
                               epoch + 1))
            if using_MAS:
                save_imgs_one_epoch(model, old_dataloader, epoch, run_root_dir)
            # Show the re_imgs of last batch and its generated imgs
        # Checkpoint
        # print('Checkpoint ...')
        if not os.path.exists(run_root_dir + '/checkpoint/'):
            os.makedirs(run_root_dir + '/checkpoint/')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, run_root_dir + '/checkpoint/checkpoint.pth.tar')

'''training for round 2'''

def train_model_round2(val_loader, epoches, model, run_root_dir, tb_writer,
                       optimizer, transforms, batch_size, cluster_num, is_simulated_dataset=False, nw=6, kmeans_mode=0,
                       transforms_cfg=None):
    acc_best = 0
    acc_sum = 0
    nmi_best = 0
    nmi_sum = 0
    clustering_times = 0
    algined_imgs_path = None
    raw_data_path = val_loader.dataset.images
    transforms_list = get_config.get_train_transformations(transforms_cfg)

    for epoch in range(epoches):
        flag_for_save_images = True
        print("\nepoch:" + str(epoch) + '\n')
        if Running_Paras.clustering_interval == 0 or epoch % Running_Paras.clustering_interval == 0:
            '''saving denoised imgs'''
            if Running_Paras.training_in_phase_2 and algined_imgs_path is not None:
                # denoised_path, labels_true = mrcdata_process.save_denoised_imgs(train_data, model, run_root_dir)
                # val_loader.dataset.images=np.array(algined_imgs_path)
                # val_loader.dataset.img_labels=labels_aligned
                train_data = torch.utils.data.DataLoader(train_data_set,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=nw,
                                                         )
                denoised_path_1, labels_true = mrcdata_process.save_denoised_imgs(val_loader, model, run_root_dir,
                                                                                  save_label='val')
                denoised_path, labels_true_1 = mrcdata_process.save_denoised_imgs(train_data, model, run_root_dir,
                                                                                  save_label='train')
                # pass
                # denoised_path,labels_true=denoised_algined_imgs_path, denoised_labels_aligned
            else:

                denoised_path, labels_true = mrcdata_process.save_denoised_imgs(val_loader, model, run_root_dir)

            # denoised_path2, labels_true2 = mrcdata_process.save_denoised_imgs(val_loader, model, run_root_dir)
            denoised_dataset = Loading_Dataset.denoised_cryoEM_Dataset_from_path(path_mrcdata=denoised_path,
                                                                                 transform=transforms.Compose(
                                                                                     [transforms.ToTensor()]))
            denoised_loader = torch.utils.data.DataLoader(denoised_dataset,
                                                          batch_size=600,
                                                          shuffle=False,
                                                          num_workers=nw,
                                                          )

            clustering_features = mrcdata_process.get_mrcs_circle_features_from_denoised_path(denoised_loader,
                                                                                              Running_Paras.circle_features_num,
                                                                                              Running_Paras.circle_feature_length)
            np.savetxt(os.path.join(run_root_dir, 'ring_features'),clustering_features)
            # clustering_features2 = mrcdata_process.get_mrcs_circle_features_from_denoised_path(denoised_loader,
            #                                                                                    Running_Paras.circle_features_num,
            #                                                                                    Running_Paras.circle_feature_length, using_spectrum_feature=False)
            # clustering_features=mrcdata_process.get_mrcs_other_features_from_denoised_path(denoised_loader,features_type='SIFT')
            '''without saving denoised imgs'''
            # mrcArrays_circle_features, labels_true= mrcdata_process.get_mrcs_circle_features_from_path(val_loader,model,
            # Running_Paras.circle_features_num,
            # Running_Paras.circle_feature_length)

            KMeans_time1 = time.time()
            if kmeans_mode == 0:
                labels_predict, cent, acc, nmi, num_class = em_k_means.features_kmeans_using_sklearn(
                    clustering_features,
                    labels_true, cluster_num)
                # labels_predict, cent, acc2, nmi2, num_class2 = em_k_means.features_kmeans_using_sklearn(
                #     clustering_features2,
                #     labels_true, cluster_num)
            elif kmeans_mode == 1:
                k_means_mrcsArrays, labels_predict, isreliable_list, cent, acc, nmi, num_class = em_k_means.circle_features_kmeans(
                    clustering_features,
                    cluster_num, is_simulated_dataset, labels_true)

            KMeans_time2 = time.time()
            print('KMeans time:' + str(KMeans_time2 - KMeans_time1))
            tb_writer.add_scalar("KMeans time:", KMeans_time2 - KMeans_time1, epoch)

            save_clustering_labels(os.path.join(run_root_dir, 'clustering_labels'), epoch, labels_predict)

            # for simulated dataset,save clustering acc
            if is_simulated_dataset:
                acc_best, nmi_best, acc_sum, nmi_sum, clustering_times = save_acc_data(acc, acc_best, nmi, nmi_best,
                                                                                       tb_writer, acc_sum, nmi_sum,
                                                                                       clustering_times, epoch,
                                                                                       run_root_dir)
                print("acc f:" + str(acc) + "\nnmi:" + str(nmi) + "\nclasses number:" + str(num_class))
                # print("acc:" + str(acc2) + "\nnmi:" + str(nmi2) + "\nclasses number:" + str(num_class2))
                #
                # tb_writer.add_scalar("accuracy2:", acc2, epoch)
                # tb_writer.add_scalar("nmi2:", nmi2, epoch)

                save_trained_model(model, optimizer, epoch, run_root_dir + '/trained_model/')
                em_k_means.clustering_res_vis(clustering_features, labels_predict, run_root_dir + '/averages/', epoch,
                                              labels_true)

            average_imgs_all, algined_imgs_path, labels_aligned, raw_align_dict = mrcdata_process.imgs_align(
                raw_data_path, labels_predict, cluster_num,
                em_k_means.get_points_nearest_to_centers(
                    clustering_features,
                    cent,
                    labels_predict), run_root_dir, 3, img_type='/raw/', true_labels=labels_true,
                raw_imgs_path=raw_data_path)
            denoised_average_imgs_all, denoised_algined_imgs_path, denoised_labels_aligned, raw_align_dict_denoised = mrcdata_process.imgs_align(
                np.asarray(denoised_path), labels_predict, cluster_num,
                em_k_means.get_points_nearest_to_centers(
                    clustering_features,
                    cent,
                    labels_predict), run_root_dir, 3, img_type='/denoised/', true_labels=labels_true)
            save_averages(average_imgs_all, average_generated_imgs=denoised_average_imgs_all, run_root_dir=run_root_dir,
                          epoch=epoch)

        if Running_Paras.training_in_phase_2 and Running_Paras.clustering_interval != 0:

            # train_data_set = Loading_Dataset.cryoEM_Dataset_from_path_p2(path_mrcdata=algined_imgs_path,
            #                                                              averages=average_imgs_all,
            #                                                              images_class=labels_aligned,
            #                                                              run_root_dir=run_root_dir,
            #                                                              isreliable_list=isreliable_list,
            #                                                              syn_rotate=True,
            #                                                              transform=transforms.Compose([
            #                                                                  transforms.ToTensor(),
            #                                                                  # transforms.Normalize(algined_raw_imgs.mean(),
            #                                                                  #                      algined_raw_imgs.std()),
            #                                                                  # transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))
            #                                                              ])
            #                                                              ,is_Normalize=False)
            # train_data_set = Loading_Dataset.cryoEM_Dataset_from_path_p2(path_mrcdata=raw_data_path,
            #                                                              averages=average_imgs_all,
            #                                                              images_class=labels_true,
            #                                                              run_root_dir=run_root_dir,
            #                                                              isreliable_list=isreliable_list,
            #                                                              raw_to_algin_dict=raw_align_dict,
            #                                                              syn_rotate=True,
            #                                                              transform=transforms.Compose([
            #                                                                  transforms.ToTensor(),
            #                                                                  # transforms.Normalize(algined_raw_imgs.mean(),
            #                                                                  #                      algined_raw_imgs.std()),
            #                                                                  # transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))
            #                                                              ])
            #                                                              ,is_Normalize=True)
            train_data_set = Loading_Dataset.cryoEM_Dataset_from_path(path_mrcdata=raw_data_path,
                                                                      path_out=run_root_dir + '/tmp/',
                                                                      # isreliable_list=isreliable_list,
                                                                      # transform=transforms.Compose(
                                                                      #    [transforms.ToTensor()]),
                                                                      transform=transforms_list,
                                                                      syn_rand_rotate=transforms.RandomRotation(
                                                                          degrees=(-180, 180)) if
                                                                      transforms_cfg[
                                                                          'is_random_rotate'] else None,
                                                                      preprocess_args=None,
                                                                      is_Normalize=transforms_cfg['is_Normalize'],
                                                                      labels_predict=labels_predict,
                                                                      isreliable_list=isreliable_list,
                                                                      raw_to_algin=raw_align_dict,
                                                                      averages=average_imgs_all
                                                                      )

            train_data = torch.utils.data.DataLoader(train_data_set,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     num_workers=nw,
                                                     )
            # 模型开始训练
            model.train(True)
            losses = AverageMeter('Loss', ':.4e')
            progress = ProgressMeter(len(train_data),
                                     [losses],
                                     prefix="Epoch: [{}]".format(epoch))
            for i, batch in enumerate(train_data):
                # img = torch.flatten(img, start_dim=1)
                img = batch['mrcdata']
                target = batch['target']
                target = Variable(target).cuda()
                is_reliable = batch['is_reliable']
                img = Variable(img).cuda()
                # forward
                gen_img, mu_img, logvar_img = model.forward(img)
                num_img = img.shape[0]
                loss = AE_Models.loss_VAE(gen_img, target, mu_img,
                                          logvar_img, 0.5) / num_img
                # loss = loss
                losses.update(loss.item())
                if i % 25 == 0:
                    progress.display(i)
                tb_writer.add_scalar("round2 loss:", loss, epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if flag_for_save_images:
                    flag_for_save_images = False
                    img_gen = torch.cat([img, target, gen_img], dim=0)
            # Get the loss of each epoch
            # record_loss_epoch = str("round2 epoch: {}, loss is {}, lr is {} \n".format((epoch + 1), loss.data.float(),
            #                                                                            optimizer.param_groups[0]['lr']))
            # # Show the loss of epoch
            # print(record_loss_epoch)
            if epoch % 1 == 0:
                if not os.path.exists(run_root_dir + 'AE_iterations/'):
                    os.makedirs(run_root_dir + 'AE_iterations/')
                    # Show the unre_imgs of last batch and its generated imgs
                save_image(to_img(img_gen),
                           run_root_dir + 'AE_iterations/' + 'round2_epoch_{}_img_gen.png'.format(
                               epoch))


def save_clustering_labels(labels_path, epoch, clustering_labels):
    if not os.path.exists(labels_path):
        os.makedirs(labels_path + '/')
    np.save(labels_path + '/predict_labels_epoch' + str(epoch) + '.npy', clustering_labels)


def save_acc_data(acc, acc_best, nmi, nmi_best, tb_writer, acc_sum, nmi_sum, clustering_times, epoch, out_path):
    if acc > acc_best:
        acc_best = acc
    if nmi > nmi_best:
        nmi_best = nmi
    acc_sum = acc_sum + acc
    nmi_sum = nmi + nmi_sum
    clustering_times = clustering_times + 1
    tb_writer.add_scalar("accuracy:", acc, epoch)
    tb_writer.add_scalar("nmi:", nmi, epoch)
    tb_writer.add_scalar("mean accuracy:", acc_sum / clustering_times,
                         clustering_times)
    tb_writer.add_scalar("mean NMI:", nmi_sum / clustering_times,
                         clustering_times)
    if not os.path.exists(out_path + 'acc_data/'):
        os.makedirs(out_path + 'acc_data/')
    with open(out_path + 'acc_data/' + 'acc_data.txt', 'w') as average_acc:
        average_acc.write('best accuracy' + str(acc_best))
        average_acc.write('\nbest NMI' + str(nmi_best))
        average_acc.write('\naverage accuracy' + str(acc_sum / clustering_times))
        average_acc.write('\naverage NMI' + str(nmi_sum / clustering_times))
    print('\naverage accuracy' + str(acc_sum / clustering_times) + '\naverage NMI' + str(nmi_sum / clustering_times))
    return acc_best, nmi_best, acc_sum, nmi_sum, clustering_times


def save_trained_model(model, optimizer, epoch, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                'epoch': epoch + 1}, save_path + 'epoch' + str(epoch) + '_model.pth.tar')


def save_averages(average_imgs_all, average_generated_imgs=None, run_root_dir=None, epoch=0):
    if not os.path.exists(run_root_dir + 'averages/'):
        os.makedirs(run_root_dir + 'averages/')
    save_image(torch.unsqueeze(torch.from_numpy(average_imgs_all), 1),
               run_root_dir + 'averages/clustering_result_' + str(epoch) + '.png')
    if average_generated_imgs is not None:
        save_image(torch.unsqueeze(torch.from_numpy(average_generated_imgs), 1),
                   run_root_dir + 'averages/generated_clustering_result_' + str(epoch) + '.png')
        # projectons_file = mrcfile.new(
        #     run_root_dir + 'averages/generated_clustering_averages_' + str(epoch) + '.mrcs',
        #     average_generated_imgs, overwrite=True)
    projectons_file = mrcfile.new(
        run_root_dir + 'averages/clustering_averages_' + str(epoch) + '.mrcs',
        average_imgs_all, overwrite=True)
    projectons_file.close()


def save_classified_particles(run_root_dir, epoch, generated_imgs, clustering_labels, class_number_arry, cluster_num):
    if not os.path.exists(run_root_dir + 'class_single_particles/epoch' + str(epoch) + '/'):
        os.makedirs(run_root_dir + 'class_single_particles/epoch' + str(epoch) + '/')
    if not os.path.exists(run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/'):
        os.makedirs(run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/')
    if Running_Paras.is_save_clustered_single_particles:
        for i in range(cluster_num):
            # class_single_particles = raw_mrcArrays[clustering_labels == i]
            generated_class_single_particles = generated_imgs[clustering_labels == i]
            generated_class_single_particles = np.squeeze(generated_class_single_particles)
            class_particles = mrcfile.new(
                run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/' + 'class_' + str(
                    i) + "_" + str(class_number_arry[i]) + '.mrcs',
                generated_class_single_particles, overwrite=True)
            class_particles.close()
        evaluate_utils.classify_mrcs(clustering_labels, cluster_num,
                                     Running_Paras.path_for_data_classify,
                                     run_root_dir + 'class_single_particles/epoch' + str(epoch) + '/')


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_imgs_one_epoch(model, old_loader, epoch, run_root_dir):
    model.eval()
    one_batch = next(iter(old_loader))
    img = one_batch['mrcdata'].cuda()
    with torch.no_grad():
        gen_img, mu_img, logvar_img = model.forward(img)
    img_contrast = torch.cat([img, gen_img], dim=0)
    if not os.path.exists(run_root_dir + 'AE_iterations/'):
        os.makedirs(run_root_dir + 'AE_iterations/')
        # Show the unre_imgs of last batch and its generated imgs

    save_image(to_img(img_contrast),
               run_root_dir + 'AE_iterations/' + 'round1_epoch_{}_img_gen_old_model.png'.format(
                   epoch + 1))

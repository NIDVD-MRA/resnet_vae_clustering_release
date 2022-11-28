# @Auther : Written/modified by Yang Yan
# @Time   : 2021/12/22 上午10:23
# @E-mail : yanyang98@yeah.net
# @Function :
from torch.autograd import Variable
import torch
import os
from torchvision.utils import save_image
from models import AE_Models, life_long_learning_models
from tqdm import tqdm
import numpy as np
from clustering import em_k_means,my_hdbscan
import mrcfile
from cryoemdata import Loading_Dataset
from cryoemdata.process import evaluate_utils, mrcdata_process
import time
import copy
from torch import nn
from  modeltrain import utils
import torchvision



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




def train_model_round1(start_epoch, epoches, model, train_data, run_root_dir, tb_writer, optimizer, old_dataloader=None, using_MAS=False ):


    if using_MAS:
        if old_dataloader is None:
            print('cannot find old dataset, MAS init failed.')
        else:
            mas= life_long_learning_models.MAS(copy.deepcopy(model), old_dataloader, torch.device('cuda'))
    else:
        mas=None
    for epoch in range(start_epoch, epoches):
        # 模型开始训练

        model.train(True)
        flag_for_save_images = True
        losses = utils.AverageMeter('Loss', ':.4e')
        progress = utils.ProgressMeter(len(train_data),
                                 [losses],
                                 prefix="Epoch: [{}]".format(epoch))
        for i, batch in enumerate(train_data):
            # img = torch.flatten(img, start_dim=1)
            img=batch['mrcdata']
            target=batch['target']
            # img = (batch['mrcdata'] - torch.min(batch['mrcdata'])) / (
            #             torch.max(batch['mrcdata']) - torch.min(batch['mrcdata']))
            # target = (batch['target'] - torch.min(batch['target'])) / (
            #             torch.max(batch['target']) - torch.min(batch['target']))
            is_reliable=batch['is_reliable']
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

            loss = model.criterion_p1(target,gen_unreliable_img)
            if mas is not None:
                loss_mas=mas.penalty(model)
                loss+=loss_mas
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
                save_image(utils.to_img(unreliable_img_gen),
                           run_root_dir + 'AE_iterations/' + 'round1_epoch_{}_unreimg_gen.png'.format(
                               epoch + 1))
            if using_MAS:
                utils.save_imgs_one_epoch(model, old_dataloader, epoch, run_root_dir)
            # Show the re_imgs of last batch and its generated imgs
        # Checkpoint
        # print('Checkpoint ...')
        if not os.path.exists(run_root_dir + '/checkpoint/'):
            os.makedirs(run_root_dir + '/checkpoint/')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, run_root_dir + '/checkpoint/checkpoint.pth.tar')
def evaluate_labels(DL,model):
    pre = []
    tru = []
    z_set=[]
    with torch.no_grad():
        for batch in DL:
            x = batch['mrcdata'].cuda()
            y = batch['label'].cuda()
            tru.append(y.cpu().numpy())
            p,z=model.predict(x)
            pre.append(p)
            z_set.append(z)
    tru = np.concatenate(tru, 0)
    pre = np.concatenate(pre, 0)
    z_set = np.concatenate(z_set, 0)
    return pre,tru,z_set


def train_model_round2(val_loader, epoches, model, run_root_dir, tb_writer,
                       optimizer, my_transforms, batch_size,cluster_num, train_data,is_simulated_dataset=False,nw=6,kmeans_mode=0):
    import pickle
    acc_best = 0
    acc_sum = 0
    nmi_best = 0
    nmi_sum = 0
    clustering_times = 0
    start_epoch=0
    if not os.path.exists(run_root_dir+'/tmp/preprocessed_data/output_denoised_tifs_path.data'):
        denoised_path, labels_true = mrcdata_process.save_denoised_imgs(val_loader, model, run_root_dir,normalize=True)
    else:
        with open(run_root_dir+'/tmp/preprocessed_data/output_denoised_tifs_path.data', 'rb') as filehandle:
            denoised_path = pickle.load(filehandle)
        with open(run_root_dir+'/tmp/preprocessed_data/output_denoised_tifs_label_path.data', 'rb') as filehandle:
            labels_true = pickle.load(filehandle)
    if os.path.exists(run_root_dir + '/checkpoint/pretrain_model.pk'):
        checkpoint = torch.load(run_root_dir + '/checkpoint/pretrain_model.pk', map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('start from phase2 epoch:'+str(start_epoch))

    denoised_dataset = Loading_Dataset.denoised_cryoEM_Dataset_from_path(path_mrcdata=denoised_path,labels=labels_true,transform=torchvision.transforms.Compose([
                    torchvision.transforms.RandomRotation(degrees=(-180, 180))]))
    denoised_loader = torch.utils.data.DataLoader(denoised_dataset,
                                                  batch_size=30,
                                                  shuffle=True,
                                                  num_workers=nw,
                                                  )
    model.decoder.add_module('sigmoid', nn.Sigmoid())
    model.pre_train(dataloader=denoised_loader,run_root_dir=run_root_dir,pre_epoch=30)
    model.encoder.log_sigma2_l.load_state_dict(model.encoder.mu_l.state_dict())
    for epoch in range(start_epoch,epoches):
        flag_for_save_images = True
        print("\nepoch:" + str(epoch) + '\n')
        # 模型开始训练
        model.train(True)
        losses = utils.AverageMeter('Loss', ':.4e')
        progress = utils.ProgressMeter(len(denoised_loader),
                                 [losses],
                                 prefix="Epoch: [{}]".format(epoch))
        for i, img in enumerate(denoised_loader):
            # img = torch.flatten(img, start_dim=1)

            # img = (batch['mrcdata']-torch.min(batch['mrcdata']))/(torch.max(batch['mrcdata'])-torch.min(batch['mrcdata']))
            img=img[0].unsqueeze(1).cuda()
            # target = (batch['target'] -torch.min(batch['target'] ))/(torch.max(batch['target'] )-torch.min(batch['target'] ))
            target=img


            # forward
            gen_img, mu_img, logvar_img = model.forward(img)
            gen_img=model.decoder.sigmoid(gen_img)
            # num_img = img.shape[0]
            loss = model.ELBO_Loss(img,gen_img)
            # loss = loss
            losses.update(loss.item())
            if i % 25 == 0:
                progress.display(i)
            tb_writer.add_scalar("round1 loss:", loss, epoch)
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
        if epoch % 2 == 0:
            if not os.path.exists(run_root_dir + 'AE_iterations/'):
                os.makedirs(run_root_dir + 'AE_iterations/')
                # Show the unre_imgs of last batch and its generated imgs
            if not os.path.exists(run_root_dir + '/checkpoint/'):
                os.makedirs(run_root_dir + '/checkpoint/')
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1}, run_root_dir + '/checkpoint/phase2_model.pk')

            save_image(utils.to_img(img_gen),
                       run_root_dir + 'AE_iterations/' + 'round2_epoch_{}_img_gen.png'.format(
                           epoch))
            labels_predict, labels_true, clustering_features = evaluate_labels(val_loader, model)


            utils.save_clustering_labels(os.path.join(run_root_dir, 'clustering_labels'), epoch, labels_predict)

            # for simulated dataset,save clustering acc
            if is_simulated_dataset:
                acc, nmi = evaluate_utils.cluster_acc(labels_predict, labels_true)
                acc_best, nmi_best, acc_sum, nmi_sum, clustering_times = utils.save_acc_data(acc, acc_best, nmi, nmi_best,
                                                                                       tb_writer, acc_sum, nmi_sum,
                                                                                       clustering_times, epoch,
                                                                                       run_root_dir)
                print("acc f:" + str(acc) + "\nnmi:" + str(nmi))
                hdbscan_labels=my_hdbscan.hdbscan_clustering(clustering_features)
                dbscan_acc, dbscan_nmi = evaluate_utils.cluster_acc(hdbscan_labels, labels_true)
                print("hdbscan acc f:" + str(dbscan_acc) + "\nhdbscan nmi:" + str(dbscan_nmi))
                utils.save_trained_model(model, optimizer, epoch, run_root_dir + '/trained_model/')
                em_k_means.clustering_res_vis(clustering_features, [labels_predict,my_hdbscan.hdbscan_clustering(clustering_features)], run_root_dir + '/averages/',
                                              epoch,
                                              labels_true)

# @Author : Written/modified by Yang Yan
# @Time   : 2022/6/20 下午7:05
# @E-mail : yanyang98@yeah.net
# @Function :
import torch
import numpy as np
from  modeltrain import utils
from torch.autograd import Variable
import torch
import os
from torchvision.utils import save_image
import Running_Paras
from models.ResNet_model import resnet18
from tqdm import tqdm
import numpy as np
from clustering import em_k_means
import mrcfile
from cryoemdata import Loading_Dataset
from cryoemdata.process import evaluate_utils, mrcdata_process
import time
import copy
from torch import nn

def simclr_train(train_loader, model, optimizer, epoch, using_pseudo_labels=False):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = utils.AverageMeter('Loss', ':.4e')
    progress = utils.ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['mrcdata']
        images_augmented = batch['target']
        # pseudo_labels=batch['pseudo labels']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)
        input_ = input_.cuda(non_blocking=True)
        # targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        if using_pseudo_labels:
            loss = SimCLRLoss(temperature=0.1)(output)
        else:
            loss=SimCLRLoss(temperature=0.1)(output)

        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)
    return loss

def train_model_round2(val_loader, epoches, VAE_model, run_root_dir, tb_writer,
                        my_transforms, batch_size, cluster_num, train_data, is_simulated_dataset=False, nw=6, kmeans_mode=0):
    from clustering.simclr_clustering_evaluate import kmeans_evaluate
    import pickle
    acc_best = 0
    acc_sum = 0
    nmi_best = 0
    nmi_sum = 0
    clustering_times = 0
    start_epoch=0
    simclr_model=resnet18(128)
    simclr_model.cuda()
    optimizer_p2 = torch.optim.SGD(simclr_model.parameters(),nesterov=False, lr=0.4, momentum=0.9, weight_decay=0.1)
    if os.path.exists(run_root_dir + '/checkpoint/phase2_simclr_model.pk'):
        checkpoint = torch.load(run_root_dir + '/checkpoint/phase2_simclr_model.pk', map_location='cpu')
        optimizer_p2.load_state_dict(checkpoint['optimizer'])
        simclr_model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('start from phase2 epoch:'+str(start_epoch))
    if not os.path.exists(run_root_dir+'/tmp/preprocessed_data/output_denoised_tifs_path.data'):
        denoised_path, labels_true = mrcdata_process.save_denoised_imgs(val_loader, VAE_model, run_root_dir, normalize=False)
    else:
        with open(run_root_dir+'/tmp/preprocessed_data/output_denoised_tifs_path.data', 'rb') as filehandle:
            denoised_path = pickle.load(filehandle)
        with open(run_root_dir+'/tmp/preprocessed_data/output_denoised_tifs_label_path.data', 'rb') as filehandle:
            labels_true = pickle.load(filehandle)
    denoised_dataset = Loading_Dataset.cryoEM_Dataset_from_path_p2(path_mrcdata=denoised_path,images_class=labels_true,transform=my_transforms)
    denoised_loader = torch.utils.data.DataLoader(denoised_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=nw,
                                                  )
    for epoch in range(start_epoch,epoches):
        print("\nepoch:" + str(epoch) + '\n')
        # 模型开始训练
        simclr_model.train(True)
        loss = simclr_train(denoised_loader, simclr_model, optimizer_p2, epoch)
        tb_writer.add_scalar("round1 loss:", loss, epoch)
        if epoch % 2 == 0:
            if not os.path.exists(run_root_dir + 'AE_iterations/'):
                os.makedirs(run_root_dir + 'AE_iterations/')
                # Show the unre_imgs of last batch and its generated imgs
            if not os.path.exists(run_root_dir + '/checkpoint/'):
                os.makedirs(run_root_dir + '/checkpoint/')
            torch.save({'optimizer': optimizer_p2.state_dict(), 'model': simclr_model.state_dict(),
                        'epoch': epoch + 1}, run_root_dir + '/checkpoint/phase2_simclr_model.pk')

            # labels_predict, labels_true, clustering_features \
            kmeans_state= kmeans_evaluate(val_loader, simclr_model,cluster_num)
            utils.save_clustering_labels(os.path.join(run_root_dir, 'clustering_labels'), epoch, kmeans_state['predicted labels'])

            # for simulated dataset,save clustering acc
            if is_simulated_dataset:
                acc, nmi = evaluate_utils.cluster_acc(kmeans_state['predicted labels'], labels_true)
                acc_best, nmi_best, acc_sum, nmi_sum, clustering_times = utils.save_acc_data(acc, acc_best, nmi, nmi_best,
                                                                                       tb_writer, acc_sum, nmi_sum,
                                                                                       clustering_times, epoch,
                                                                                       run_root_dir)
                print("acc f:" + str(acc) + "\nnmi:" + str(nmi))

                utils.save_trained_model(simclr_model, optimizer_p2, epoch, run_root_dir + '/trained_model/')
                em_k_means.clustering_res_vis(kmeans_state['features'], kmeans_state['predicted labels'], run_root_dir + '/averages/',
                                              epoch,
                                              labels_true)


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR
        """

        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss

# @Author : Written/modified by Yang Yan
# @Time   : 2022/6/16 下午10:44
# @E-mail : yanyang98@yeah.net
# @Function :
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from models.AE_Models import ResNet18Enc,ResNet18Dec
import numpy as np
import os
from tqdm import tqdm
from torch.optim import Adam
import itertools
from sklearn.mixture import GaussianMixture
from torchvision.utils import save_image
from clustering import em_k_means,my_hdbscan

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
    return sum([w[i,j] for i,j in zip(ind[0],ind[1])])*1.0/Y_pred.size, w


class ResVaDE(nn.Module):
    def __init__(self,args):
        super(ResVaDE,self).__init__()
        self.encoder=ResNet18Enc(z_dim=args.hid_dim)
        self.decoder=ResNet18Dec(z_dim=args.hid_dim)

        self.pi_=nn.Parameter(torch.FloatTensor(args.nClusters,).fill_(1)/args.nClusters,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(args.nClusters,args.hid_dim).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(args.nClusters,args.hid_dim).fill_(0),requires_grad=True)


        self.args=args

    def criterion_p1(self, x, x_, *params):
        # Loss= torch.nn.L1Loss(reduction='sum')
        # Loss = nn.MSELoss(reduction='sum')
        Loss = nn.MSELoss()
        return Loss(x, x_)
    def forward(self,x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    def pre_train(self,dataloader,pre_epoch=5,run_root_dir='./'):
        start_epoch=0
        opti = Adam(self.parameters(), lr=1e-4)
        if os.path.exists(run_root_dir+ '/checkpoint/pretrain_model.pk'):
            checkpoint = torch.load(run_root_dir+ '/checkpoint/pretrain_model.pk', map_location='cpu')
            opti.load_state_dict(checkpoint['optimizer'])
            self.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch']
            print('start from pretraining epoch:'+str(start_epoch))
        # if  not os.path.exists(run_root_dir+ '/checkpoint/pretrain_model.pk'):

            # Loss=nn.MSELoss(reduction='sum')
            # opti=Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))


        print('Pretraining......')
        epoch_bar=tqdm(range(start_epoch,pre_epoch))
        for epoch in epoch_bar:
            L=0
            for x,y in dataloader:
                if self.args.cuda:
                    x=x.unsqueeze(1).cuda()

                z,_=self.encoder(x)
                x_=self.decoder.sigmoid(self.decoder(z))
                # loss=Loss(x,x_)
                loss=self.criterion_p1(x, x_)

                L+=loss.detach().cpu().numpy()

                opti.zero_grad()
                loss.backward()
                opti.step()

            epoch_bar.write('L2={:.4f}'.format(L/len(dataloader)))

            if epoch % 3 == 0:
                labels_predict, labels_true, clustering_features = self.evaluate_labels(dataloader)
                if not os.path.exists(run_root_dir + 'AE_iterations/'):
                    os.makedirs(run_root_dir + 'AE_iterations/')
                    # Show the unre_imgs of last batch and its generated imgs
                # if x_ != 0:
                save_image(torch.cat([x,x_], dim=0),
                           run_root_dir + 'AE_iterations/' + 'pretrain_epoch_{}_gen.png'.format(
                               epoch + 1))
                hdbscan_labels=my_hdbscan.hdbscan_clustering(clustering_features,min_size=5)
                em_k_means.clustering_res_vis(clustering_features,
                                              [labels_predict, hdbscan_labels],
                                              run_root_dir + '/averages/',
                                              epoch,
                                              labels_true)
                print('HDBSCAN Acc={:.4f}%'.format(cluster_acc(hdbscan_labels, labels_true)[0] * 100))
                if not os.path.exists(run_root_dir + '/checkpoint/'):
                    os.makedirs(run_root_dir + '/checkpoint/')
                torch.save({'optimizer': opti.state_dict(), 'model': self.state_dict(),
                            'epoch': epoch + 1}, run_root_dir+ '/checkpoint/pretrain_model.pk')

        self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

        Z = []
        Y = []
        with torch.no_grad():
            for x, y in dataloader:
                if self.args.cuda:
                    x = x.unsqueeze(1).cuda()

                z1, z2 = self.encoder(x)
                # m=F.mse_loss(z1, z2)
                assert F.mse_loss(z1, z2) == 0
                Z.append(z1)
                Y.append(y)

        Z = torch.cat(Z, 0).detach().cpu().numpy()
        Y = torch.cat(Y, 0).detach().numpy()

        gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')

        pre = gmm.fit_predict(Z)
        print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

        self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
        self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
        self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())



        # else:
        #     checkpoint = torch.load(run_root_dir+ '/checkpoint/pretrain_model.pk', map_location='cpu')
        #     # optimizer.load_state_dict(checkpoint['optimizer'])
        #     self.load_state_dict(checkpoint['model'])
        #     self.load_state_dict(checkpoint)




    def predict(self,x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))

        yita=yita_c.detach().cpu().numpy()
        return np.argmax(yita,axis=1),z.detach().cpu().numpy()


    def ELBO_Loss(self,x,x_pro,L=1):
        det=1e-10

        L_rec=0

        z_mu, z_sigma2_log = self.encoder(x)
        for l in range(L):

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            # x_pro=self.decoder.sigmoid(self.decoder(z))

            L_rec+=F.binary_cross_entropy(x_pro,x)

        L_rec/=L

        Loss=L_rec*x.size(3)

        pi=self.pi_
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        # pdfs=self.gaussian_pdfs_log(z,mu_c,log_sigma2_c)
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


        return Loss

    def evaluate_labels(self,DL):
        Z = []
        Y = []
        with torch.no_grad():
            for x, y in DL:
                if self.args.cuda:
                    x = x.unsqueeze(1).cuda()

                z1, z2 = self.encoder(x)
                # m=F.mse_loss(z1, z2)
                # assert F.mse_loss(z1, z2) == 0
                Z.append(z1)
                Y.append(y)

        Z = torch.cat(Z, 0).detach().cpu().numpy()
        Y = torch.cat(Y, 0).detach().numpy()

        gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')

        pre = gmm.fit_predict(Z)
        print('Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

        return pre, Y, Z

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.args.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)



    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std).cuda()
        return epsilon * std + mean
    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))



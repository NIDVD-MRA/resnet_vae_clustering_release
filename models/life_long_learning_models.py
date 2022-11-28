# @Author : Written/modified by Yang Yan
# @Time   : 2022/6/6 下午5:07
# @E-mail : yanyang98@yeah.net
# @Function :
import torch.nn as nn
import torch
import numpy as np

class MAS(object):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """

    def __init__(self, model: nn.Module, dataloader, device,old_omega=None):
        self.model = model
        self.dataloader = dataloader
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.device = device
        self._precision_matrices = self._calculate_omega()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _calculate_omega(self):
        print('Computing MAS')

        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        dataloader_num = len(self.dataloader)
        # num_data = sum([len(loader) for loader in self.dataloaders])
        # for dataloader in self.dataloaders:
        for data in self.dataloader:
            self.model.zero_grad()
            reconstruct,mean,var = self.model(data['mrcdata'][0].unsqueeze(0).to(self.device))
            reconstruct.pow_(2)
            mean.pow_(2)
            var.pow_(2)
            loss = torch.sum(reconstruct, dim=1)
            loss = loss.mean()+mean.mean()+var.mean()
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.abs() / dataloader_num
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        # new_paras= {n: p for n, p in model.named_parameters() if p.requires_grad}
        # old=self.model.named_parameters()
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
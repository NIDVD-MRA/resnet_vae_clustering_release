# @Author : Written/modified by Yang Yan
# @Time   : 2021/12/15 上午11:09
# @E-mail : yanyang98@yeah.net
# @Function : main program.
import os
import sys
import time
import warnings
import torch
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from cryoemdata import Loading_Dataset
import Running_Paras
from models import AE_Models
from modeltrain import model_train_VAE
import yaml
from easydict import EasyDict
import get_config


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 运行设置
sys.stdout = Logger('result/a.log', sys.stdout)
sys.stderr = Logger('result/a.log_file', sys.stderr)
warnings.filterwarnings('ignore')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1,2,3'

if __name__ == "__main__":
    '''get config'''
    cfg = EasyDict()
    with open('settings.yml', 'r') as stream:
        config = yaml.safe_load(stream)
    for k, v in config.items():
        cfg[k] = v

    ''' Set the running config of this experiment '''
    time_of_run = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print(time_of_run)
    run_root_dir = cfg['path_result_dir']
    # old_model_path=cfg['old_model_path']
    path_data = cfg['path_data']
    epoches_round1 = cfg['epoches_round1']
    cluster_num = cfg['cluster_num']
    epoches_round2=cfg['epoches_round2']
    tb_writer = SummaryWriter(log_dir=run_root_dir + "tensorboard/")
    if not os.path.exists(run_root_dir):
        os.makedirs(run_root_dir)
    '''save the config settings'''
    with open(run_root_dir + 'settings.txt', 'w') as settings:
        settings.write('dataset path:' + path_data + '\n')
        settings.write('var:' + str(Running_Paras.var) + '\n')
        settings.write('clustering_interval:' + str(Running_Paras.clustering_interval) + '\n')
        settings.write('clustering number:' + str(cluster_num) + '\n')
        settings.write('epoches_round1:' + str(epoches_round1) + '\n')
        settings.write('epoches_round2:' + str(Running_Paras.epoches_round2) + '\n')
        if Running_Paras.using_circle_features:
            settings.write('using circle features' + '\n')

    '''prepare the dataset'''
    batch_size = cfg['batch_size']
    nw = min(os.cpu_count(), 60)  # number of workers
    print('Using {} dataloader workers'.format(nw))
    #
    # if Running_Paras.training_in_phase_2:
    #     transforms_list=[transforms.ToTensor(),transforms.RandomRotation(degrees=(-180, 180)),]
    #     # transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))
    # else:
    #     transforms_list = [transforms.ToTensor(),transforms.RandomRotation(degrees=(-180, 180)), ]
    transforms_list = get_config.get_train_transformations(cfg['augmentation_kwargs']['phase 1'])

    train_dataset = Loading_Dataset.cryoEM_Dataset_from_path(path_mrcdata=path_data,
                                                             path_out=run_root_dir + '/tmp/',
                                                             # isreliable_list=isreliable_list,
                                                             transform=transforms_list,
                                                             syn_rand_rotate=transforms.RandomRotation(
                                                                 degrees=(-180, 180)) if
                                                             cfg['augmentation_kwargs']['phase 1'][
                                                                 'is_random_rotate'] else None,
                                                             preprocess_args=cfg['preprocess_kwargs'],
                                                             is_Normalize=cfg['augmentation_kwargs']['phase 1'][
                                                                 'is_Normalize']
                                                             )

    val_dataset = Loading_Dataset.cryoEM_Dataset_from_path(path_mrcdata=path_data,
                                                           path_out=run_root_dir + '/tmp/',
                                                           # isreliable_list=isreliable_list,
                                                           transform=transforms.Compose([
                                                               transforms.ToTensor(),
                                                               # transforms.RandomRotation(degrees=(-180, 180))
                                                               # transforms.RandomRotation(degrees=(-180, 180)),

                                                               # transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))
                                                           ]),
                                                           syn_rand_rotate=transforms.RandomRotation(
                                                               degrees=(-180, 180)) if
                                                           cfg['augmentation_kwargs']['phase 1'][
                                                               'is_random_rotate'] else None,
                                                           is_Normalize=cfg['augmentation_kwargs']['phase 1'][
                                                               'is_Normalize']
                                                           )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=nw,
                                                   )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=nw,
                                                 )
    # For calculating the time cost
    time_start_phase1 = time.time()
    '''build the model '''
    # Build the model and put it on cuda
    model = AE_Models.VAE_RESNET18(z_dim=128)
    if torch.cuda.is_available():
        # model = nn.DataParallel(model)
        model.cuda()
        # model = nn.DataParallel(model)
        print("Is model on gpu: {}".format(next(model.parameters()).is_cuda))
    print(model)
    # from torchsummary import summary
    # summary(model, (1, 128, 128))

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
    # weight_decay = 1e-5
    # epoches_round1 = epoches_round1
    # epoches_round2 = Running_Paras.epoches_round2
    # epoches_round2 = Running_Paras.epoches_round2
    # clustering_times = 0
    # stdVar = 0
    # acc_best = 0
    # acc_sum = 0
    lr_p1 = 1e-3
    lr_p2= 1e-5
    optimizer_p1 = optim.Adam(model.parameters(), lr=lr_p1)
    optimizer_p2 = optim.Adam(model.parameters(), lr=lr_p2)

    '''if using Continual Learning...'''
    old_dataloader= None
    # Checkpoint
    if os.path.exists(run_root_dir + '/checkpoint/checkpoint.pth.tar'):
        print('Restart from checkpoint ' + run_root_dir + '/checkpoint/checkpoint.pth.tar')
        checkpoint = torch.load(run_root_dir + '/checkpoint/checkpoint.pth.tar', map_location='cpu')
        optimizer_p1.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        # model = torch.nn.DataParallel(model)
        # model.to(device)
        start_epoch = checkpoint['epoch']
    # else:
    #     print('No checkpoint file at {}' + '/checkpoint/')
    #     start_epoch = 0
    #     # model = model.to(device)
    elif 'old_proj_path' in cfg and cfg['start_from_old_model']:
        old_proj_path = cfg['old_proj_path']
        print('Get old model ' + old_proj_path + '/checkpoint/checkpoint.pth.tar')
        old_model_checkpoint = torch.load(old_proj_path + '/checkpoint/checkpoint.pth.tar', map_location='cpu')
        # optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_p1.load_state_dict(old_model_checkpoint['optimizer'])
        model.load_state_dict(old_model_checkpoint['model'])
        # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        # model = torch.nn.DataParallel(model)
        # model.to(device)
        start_epoch = old_model_checkpoint['epoch']
        start_epoch = 0
        # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model'].items()})
        # model = torch.nn.DataParallel(model)
        # model.to(device)
        # start_epoch = checkpoint['epoch']
        if 'old_data_path' in cfg:
            old_data_path = cfg['old_data_path']
        else:
            old_data_path =None
        old_dataset = Loading_Dataset.cryoEM_Dataset_from_path(path_mrcdata=old_data_path,
                                                               path_out=old_proj_path + '/tmp/',
                                                               # isreliable_list=isreliable_list,
                                                               transform=transforms.Compose([
                                                                   transforms.ToTensor(),
                                                                   # transforms.RandomRotation(degrees=(-180, 180))
                                                                   # transforms.RandomRotation(degrees=(-180, 180)),

                                                                   # transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))
                                                               ]),
                                                               syn_rand_rotate=transforms.RandomRotation(
                                                                   degrees=(-180, 180)) if
                                                               cfg['augmentation_kwargs']['phase 1']['is_random_rotate'] else None,
                                                               is_Normalize=cfg['augmentation_kwargs']['phase 1']['is_Normalize']
                                                               )
        old_dataloader = torch.utils.data.DataLoader(old_dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=nw,
                                                     )
    else:
        print('No old model file used at {}' + '/checkpoint/')
        start_epoch = 0




        # model = model.to(device)
    '''train the model'''
    model_train_VAE.train_model_round1(start_epoch, epoches_round1, model, train_dataloader, run_root_dir, tb_writer,
                                       optimizer_p1, old_dataloader=old_dataloader, using_MAS=cfg['using_mas'])

    time_end_phase1 = time.time()
    time_cost_phase1 = time_end_phase1 - time_start_phase1
    with open(run_root_dir + 'time_cost.txt', 'w') as time_cost:
        time_cost.write('phase 1:' + str(time_cost_phase1) + 's\n')

    model_train_VAE.train_model_round2(val_dataloader, epoches_round2, model,
                                       run_root_dir, tb_writer, optimizer_p2, transforms, batch_size, cluster_num=cluster_num,
                                       is_simulated_dataset=cfg['is_simulated_dataset'], nw=nw,
                                       kmeans_mode=cfg['kmeans_mode'],transforms_cfg=cfg['augmentation_kwargs']['phase 2'])

    # For calculating the time cost

    time_end_phase2 = time.time()
    time_cost_phase2 = time_end_phase2 - time_end_phase1
    time_cost_all = time_end_phase2 - time_start_phase1
    with open(run_root_dir + 'time_cost.txt', 'w') as time_cost:
        time_cost.write('phase 1:' + str(time_cost_phase1) + 's\n')
        time_cost.write('phase 2:' + str(time_cost_phase2) + 's\n')
        time_cost.write('all:' + str(time_cost_all) + 's\n')
    print("The expriment cost " + str(time_cost_all) + "s")

    # model_test.test_with_trained_model(Running_Paras.is_simulated_dataset, Running_Paras.path_data, run_root_dir,
    #                                    run_root_dir + '/trained_model/' + 'epoch' + str(
    #                                        Running_Paras.epoches_round2) + '_model.pth.tar', 10, 10)

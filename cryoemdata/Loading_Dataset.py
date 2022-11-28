import os
import pickle
import random

import mrcfile
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from cryoemdata.process import mrcdata_preprocess
import Running_Paras


class Dataset_phase1(Dataset):
    """自定义数据集"""

    def __init__(self, images, images_class, isreliable_list, transform=None):
        self.images = images
        self.img_labels = images_class
        self.transform = transform
        self.isreliable_list = isreliable_list

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        mrcdata = self.images[item]
        mrcdata = np.require(mrcdata, requirements=['W'])
        mrcdata = Image.fromarray(mrcdata)
        if self.transform is not None:
            mrcdata = self.transform(mrcdata)
        # mrcdata=torch.unsqueeze(mrcdata,dim=0)
        label = self.img_labels[item]
        is_reliable = self.isreliable_list[item]
        return mrcdata, label, is_reliable

    def __getitem__(self, item):
        mrcdata = self.images[item]
        label = self.img_labels[item]
        is_reliable = self.isreliable_list[item]
        target = mrcdata
        mrcdata = np.require(mrcdata, requirements=['W'])
        mrcdata = Image.fromarray(mrcdata)
        target = Image.fromarray(target)
        if self.transform is not None:
            mrcdata = self.transform(mrcdata)
            target = self.transform(target)
            if Running_Paras.training_in_phase_2:
                random_degree = random.randint(0, 360)
                mrcdata = transforms.functional.rotate(mrcdata, random_degree)
                target = transforms.functional.rotate(target, random_degree)

        return mrcdata, target, label, is_reliable


class Dataset_phase2(Dataset):
    """自定义数据集"""

    def __init__(self, images, averages, images_class, isreliable_list, transform=None):
        self.images = images
        self.img_labels = images_class
        self.transform = transform
        self.isreliable_list = isreliable_list
        self.averages = averages

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        mrcdata = self.images[item]
        label = self.img_labels[item]
        is_reliable = self.isreliable_list[item]
        if is_reliable:
            target = self.averages[label]
        else:
            target = mrcdata
        mrcdata = np.require(mrcdata, requirements=['W'])
        mrcdata = Image.fromarray(mrcdata)
        target = Image.fromarray(target)
        if self.transform is not None:
            mrcdata = self.transform(mrcdata)
            target = self.transform(target)
            random_degree = random.randint(0, 360)
            mrcdata = transforms.functional.rotate(mrcdata, random_degree)
            target = transforms.functional.rotate(target, random_degree)

        # mrcdata=torch.unsqueeze(mrcdata,dim=0)

        return mrcdata, target, label, is_reliable


class Dataset_mrc(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        mrcdata = mrcfile.open(self.images_path[item]).data
        mrcdata = np.require(mrcdata, requirements=['W'])
        img = Image.fromarray(mrcdata)

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'F':
            raise ValueError("image: {} isn't F mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class cryoEM_Dataset_from_path(Dataset):
    """自定义数据集"""

    def __init__(self, path_mrcdata, path_out, isreliable_list=None, transform=None, syn_rand_rotate=None,
                 preprocess_args=None, is_Normalize=None,normal_scale=30,labels_predict=None,raw_to_algin=None,averages=None):
        path_out += '/preprocessed_data/'
        if not os.path.exists(path_out + '/output_tifs_path.data'):
            mrcdata_preprocess.get_resized_cropped_single_images(path_mrcdata, path_out, preprocess_args['resize'],
                                                                 preprocess_args[
                                                                     'crop_ratio'] if 'crop_ratio' in preprocess_args else None,is_norm=True)
        with open(path_out + '/output_tifs_path.data', 'rb') as filehandle:
            path_data = pickle.load(filehandle)
        with open(path_out + '/output_tifs_label.data', 'rb') as filehandle:
            label_data = pickle.load(filehandle)
        with open(path_out + '/means_stds.data', 'rb') as filehandle:
            means, stds = pickle.load(filehandle)
        self.images = np.asarray(path_data)
        self.img_labels = label_data
        self.labels_p=labels_predict
        self.isnorm=is_Normalize
        self.mean_std=[means,stds]
        self.normal_scale=normal_scale
        self.raw_to_algin=raw_to_algin
        # if is_Normalize:
        #     self.transform = transforms.Compose(
        #         transform.transforms + [transforms.Normalize(means, stds)])
        # else:
        #     self.transform=transform
        self.transform=transform
        self.isreliable_list = isreliable_list
        self.rand_rotate = syn_rand_rotate
        self.averages=averages

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        tif_path = self.images[item]
        mrcdata = Image.open(tif_path)
        label = self.img_labels[item]
        if not self.isreliable_list:
            is_reliable = False
        else:
            is_reliable = self.isreliable_list[item]
        # target = mrcdata
        # mrcdata = np.require(mrcdata, requirements=['W'])
        # mrcdata = Image.fromarray(mrcdata)
        # target = Image.fromarray(target)
        if self.rand_rotate is not None:
            mrcdata = self.rand_rotate(mrcdata)
            # target = self.transform(target)

        if is_reliable:
            mrcdata = Image.open(self.raw_to_algin[tif_path])
            label_p = self.labels_p[item]
            target = self.averages[label_p]
        else:
            target = mrcdata

        if self.isnorm is not None:
            mrcdata=  (mrcdata - np.min(mrcdata)) * (self.normal_scale / (np.max(mrcdata) - np.min(mrcdata)))
            mrcdata = mrcdata - np.mean(mrcdata)
            target=  (target - np.min(target)) * (self.normal_scale / (np.max(target) - np.min(target)))
            target = target - np.mean(target)
        # if Running_Paras.training_in_phase_2:
        #     random_degree = random.randint(0, 360)
        #     mrcdata = transforms.functional.rotate(mrcdata, random_degree)
        #     target = transforms.functional.rotate(target, random_degree)
        if self.transform is not None:
            mrcdata = self.transform(mrcdata)
        target = transforms.ToTensor()(target)
        out = {'mrcdata': mrcdata, 'target': target,'label': label,
               'is_reliable': is_reliable,'path':tif_path}
        return out


class cryoEM_Dataset_from_path_p2(Dataset):
    """自定义数据集"""

    def __init__(self, path_mrcdata, images_class, raw_to_algin_dict,run_root_dir=None, averages=None, isreliable_list=None, transform=None, is_Normalize=None, syn_rotate=False,norm_scale=30):
        self.images = path_mrcdata
        self.img_labels = images_class
        self.isreliable_list = isreliable_list
        self.averages = averages
        self.syn_rotate=syn_rotate
        self.normal_scale = norm_scale
        self.transform = transform
        self.isnorm = is_Normalize
        self.raw_to_algin=raw_to_algin_dict
        if is_Normalize:
            with open(run_root_dir + '/tmp/preprocessed_data/means_stds.data', 'rb') as filehandle:
                means, stds = pickle.load(filehandle)

            self.mean_std = [means, stds]
        #     self.transform = transforms.Compose(
        #         transform.transforms + [transforms.Normalize(means, stds)])
        # else:
        #     self.transform=transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        if self.isreliable_list is None:
            is_reliable = False
        else:
            is_reliable = self.isreliable_list[item]

        tif_path = self.images[item]
        if is_reliable:
            mrcdata=Image.open(self.raw_to_algin[tif_path])
        else:
            mrcdata = Image.open(tif_path)
        label = self.img_labels[item]


        if is_reliable:

            if self.isnorm:
                target = (self.averages[label]-np.mean(self.averages))/np.std(self.averages)
            else:
                target=self.averages[label]
        else:
            target = mrcdata

        if self.syn_rotate:
            random_degree = random.randint(0, 360)
            mrcdata = transforms.functional.rotate(mrcdata, random_degree)
            target = transforms.functional.rotate(Image.fromarray(np.array(target)), random_degree)

        if self.isnorm:
            if self.isnorm is not None:
                mrcdata = (mrcdata - np.min(mrcdata)) * (self.normal_scale / (np.max(mrcdata) - np.min(mrcdata)))
                mrcdata = mrcdata - np.mean(mrcdata)
                target = (target - np.min(target)) * (self.normal_scale / (np.max(target) - np.min(target)))
                mrcdata = mrcdata - np.mean(target)

        # mrcdata = np.require(mrcdata, requirements=['W'])
        # mrcdata = Image.fromarray(mrcdata)

        if isinstance(target, np.ndarray):
            target = Image.fromarray(target)
        if self.transform is not None:
            mrcdata = self.transform(mrcdata)
            target = self.transform(target)

        # if self.isnorm is not None:
        #     target=(target-self.mean_std[0])/self.mean_std[1]
        # mrcdata=torch.unsqueeze(mrcdata,dim=0)

        out = {'mrcdata': mrcdata, 'target': target,'label': label,
               'is_reliable': is_reliable,'path':tif_path}
        return out


class denoised_cryoEM_Dataset_from_path(Dataset):
    """自定义数据集"""

    def __init__(self, path_mrcdata, labels=None,transform=None):
        self.images = path_mrcdata
        self.labels=labels
        self.transform=transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        tif_path = self.images[item]

        mrcdata = (Image.open(tif_path))
        if self.transform is not None:
            mrcdata=np.asarray(self.transform(mrcdata))

        if self.labels is not None:
            label = self.labels[item]
            return mrcdata,label
        else:
            return mrcdata

class denoised_cryoEM_Dataset_from_path_simclr(Dataset):

    def __init__(self, images, averages, images_class, run_root_dir, isreliable_list=None, my_transform=None, is_Normalize=None):
        self.images = images
        self.img_labels = images_class
        self.isreliable_list = isreliable_list
        self.averages = averages
        with open(run_root_dir + '/tmp/preprocessed_data/means_stds.data', 'rb') as filehandle:
            means, stds = pickle.load(filehandle)
        if is_Normalize:
            self.my_transform = transforms.Compose(
                my_transform.transforms + [transforms.Normalize(means, stds)])
        else:
            self.my_transform=my_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        tif_path = self.images[item]
        mrcdata = Image.open(tif_path)
        label = self.img_labels[item]
        if not self.isreliable_list:
            is_reliable = False
        else:
            is_reliable = self.isreliable_list[item]
        if is_reliable:
            target = self.averages[label]
        else:
            target = mrcdata
        # mrcdata = np.require(mrcdata, requirements=['W'])
        # mrcdata = Image.fromarray(mrcdata)
        if isinstance(target, np.ndarray):
            target = Image.fromarray(target)
        if self.my_transform is not None:
            mrcdata = self.my_transform(mrcdata)
            target = self.my_transform(target)

            mrcdata = transforms.RandomRotation([-180,180])(mrcdata)
            target = transforms.RandomRotation(target)

        # mrcdata=torch.unsqueeze(mrcdata,dim=0)
        out = {'mrcdata': mrcdata, 'target': target,'label': label,
               'is_reliable': is_reliable}
        return out


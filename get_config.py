# @Author : Written/modified by Yang Yan
# @Time   : 2022/4/28 下午10:30
# @E-mail : yanyang98@yeah.net
# @Function :
import torchvision.transforms as transforms
from torchtoolbox.transform import Cutout
import numpy as np
import random
import math
import PIL


class rondom_pixel_lost(object):

    def __init__(self, p=0.5, ratio=(0.4, 1 / 0.4)):

        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.ratio = ratio

    def __call__(self, img):
        img = np.array(img)
        if random.random() < self.p:
            img_h, img_w = img.shape
            lost_pixel_num = int(img.size * self.ratio)
            mask = np.concatenate((np.zeros(lost_pixel_num), np.ones(img.size - lost_pixel_num)), axis=0)
            np.random.shuffle(mask)
            mask = mask.reshape((img_w, img_h))
            img = img * mask
        return PIL.Image.fromarray(img)


def get_train_transformations(p):
    my_transform = transforms.Compose([])

        # Standard augmentation strategy

        # my_transform2= transforms.Compose([
        #     transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
        #     # transforms.RandomHorizontalFlip(),
        #
        #     # transforms.RandomRotation(degrees=(-180, 180)),
        #
        #     Cutout(p['augmentation_kwargs']['cutout_kwargs']['p'],value=p['augmentation_kwargs']['cutout_kwargs']['value']),
        #
        #     transforms.ToTensor()
        # ])
    if 'random_resized_crop' in p:
        my_transform = transforms.Compose(
            my_transform.transforms + [
                transforms.RandomResizedCrop(**p['random_resized_crop'])])

    if 'cutout_kwargs' in p:
        my_transform = transforms.Compose(
            my_transform.transforms + [Cutout(p['cutout_kwargs']['p'],
                                              value=p['cutout_kwargs']['value'])])

    if 'random_pixel_lost' in p:
        my_transform = transforms.Compose(
            my_transform.transforms + [rondom_pixel_lost(p['random_pixel_lost']['p'],
                                                         ratio=p['random_pixel_lost'][
                                                             'ratio'])])

    if 'random_rotate' in p:
        my_transform = transforms.Compose(
            my_transform.transforms + [transforms.RandomRotation(p['random_rotate']['degrees'])])
    my_transform = transforms.Compose(
        my_transform.transforms + [transforms.ToTensor()])
    return my_transform
# def get_old_model(p):

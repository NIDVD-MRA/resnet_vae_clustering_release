# @Author : Written/modified by Yang Yan
# @Time   : 2022/6/7 下午4:20
# @E-mail : yanyang98@yeah.net
# @Function :

import numpy as np
import random
import math
import PIL


class rondom_pixel_lost(object):

    def __init__(self, p=0.5,  ratio=(0.4, 1 / 0.4)):

        if p < 0 or p > 1:
            raise ValueError("range of random erasing probability should be between 0 and 1")
        self.p = p
        self.ratio = ratio
    def __call__(self, img):
        img = np.array(img)
        if random.random() < self.p:
            img_h, img_w = img.shape
            lost_pixel_num=int(img.size*self.ratio)
            mask = np.concatenate((np.zeros(lost_pixel_num), np.ones(img.size - lost_pixel_num)), axis=0)
            np.random.shuffle(mask)
            mask = mask.reshape((img_w, img_h))
            img=img*mask
        return PIL.Image.fromarray(img)

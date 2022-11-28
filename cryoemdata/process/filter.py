import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm
from math import sqrt
import mrcdata_process
import Running_Paras
from torchvision.utils import save_image

# kernel_5 = np.array([[0.003, 0.0133, 0.0219, 0.133, 0.003], [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
#                      [0.0219, 0.0983, 0.1621, 0.0983, 0.0219], [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
#                      [0.003, 0.0133, 0.0219, 0.0133, 0.003]])
# kernel_7 = np.array(
#     [[0.0166, 0.0184, 0.0195, 0.0199, 0.0195, 0.0184, 0.0166], [0.0184, 0.0203, 0.0216, 0.0220, 0.0216, 0.0203, 0.0184],
#      [0.0195, 0.0216, 0.0229, 0.0234, 0.0229, 0.0216, 0.0195], [0.0199, 0.0220, 0.0234, 0.0238, 0.0234, 0.0220, 0.0199],
#      [0.0195, 0.0216, 0.0229, 0.0234, 0.0229, 0.0216, 0.0195], [0.0184, 0.0203, 0.0216, 0.0220, 0.0216, 0.0203, 0.0184],
#      [0.0166, 0.0184, 0.0195, 0.0199, 0.0195, 0.0184, 0.0166]])
# raw_mrcArrays, labels_true = mrcdata_process.loading_generated_projections(Running_Paras.path_noisy_data)
# mrc = raw_mrcArrays[0]
# filtered_mrc = ndimage.convolve(mrc, kernel_5)
# filtered_mrc = np.expand_dims(filtered_mrc, axis=0)
# filtered_mrc_7 = ndimage.convolve(mrc, kernel_7)
# filtered_mrc_7 = np.expand_dims(filtered_mrc_7, axis=0)
# mrc = np.expand_dims(mrc, axis=0)
# filtered_mrc = np.append(mrc, filtered_mrc, axis=0)
# filtered_mrc = np.append(filtered_mrc, filtered_mrc_7,axis=0)
# save_image(torch.unsqueeze(torch.from_numpy(filtered_mrc), 1), 'mrc.jpg')

def gauss_filter_7(raw_mrc_array):
    kernel_7 = np.array(
        [[0.0166, 0.0184, 0.0195, 0.0199, 0.0195, 0.0184, 0.0166],
         [0.0184, 0.0203, 0.0216, 0.0220, 0.0216, 0.0203, 0.0184],
         [0.0195, 0.0216, 0.0229, 0.0234, 0.0229, 0.0216, 0.0195],
         [0.0199, 0.0220, 0.0234, 0.0238, 0.0234, 0.0220, 0.0199],
         [0.0195, 0.0216, 0.0229, 0.0234, 0.0229, 0.0216, 0.0195],
         [0.0184, 0.0203, 0.0216, 0.0220, 0.0216, 0.0203, 0.0184],
         [0.0166, 0.0184, 0.0195, 0.0199, 0.0195, 0.0184, 0.0166]])
    num=raw_mrc_array.shape[0]
    for i in range(num):
        filtered_mrc=ndimage.convolve(raw_mrc_array[i],kernel_7)
        raw_mrc_array[i]=filtered_mrc
    return raw_mrc_array

def gauss_filter_5(raw_mrc_array):
    kernel_7 = np.array([[0.003, 0.0133, 0.0219, 0.133, 0.003], [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                     [0.0219, 0.0983, 0.1621, 0.0983, 0.0219], [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                     [0.003, 0.0133, 0.0219, 0.0133, 0.003]])
    num=raw_mrc_array.shape[0]
    for i in range(num):
        filtered_mrc=ndimage.convolve(raw_mrc_array[i],kernel_7)
        raw_mrc_array[i]=filtered_mrc
    return raw_mrc_array


def cal_distance(pa, pb):
    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
    return dis

def make_transform_matrix(image,d):
    transfor_matrix = np.zeros([image.shape[0], image.shape[1]])
    center_point = tuple(map(lambda x: (x - 1) / 2, image.shape))
    for i in range(transfor_matrix.shape[0]):
        for j in range(transfor_matrix.shape[1]):
            dis = cal_distance(center_point, (i, j))
            transfor_matrix[i, j] = np.exp(-(dis ** 2) / (2 * (d ** 2)))
    return transfor_matrix

def GaussianLowFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    d_matrix = make_transform_matrix(image,d)
    new_img =(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img

def gauss_low_pass_filter(raw_mrc_array,d):
    num=raw_mrc_array.shape[0]
    phbr=tqdm(range(num))
    phbr.set_description("filtering")
    for i in phbr:
        raw_mrc_array[i]=GaussianLowFilter(raw_mrc_array[i],d)
    return raw_mrc_array

# a=1
# b=type(a)
# print(b)
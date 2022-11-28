# @Author : Written/modified by Yang Yan
# @Time   : 2022/4/8 下午8:39
# @E-mail : yanyang98@yeah.net
# @Function : generate images suitable for network training
import mrcfile
import numpy as np
import PIL.Image
import os
from tqdm import tqdm
import multiprocessing
from functools import partial
import pickle
import random
from PIL import Image
import math

def mrcs_resize(mrcs, width, height,is_norm=False):
    resized_mrcs = np.zeros((mrcs.shape[0], width, height))
    # pbar = tqdm(range(mrcs.shape[0]))
    # pbar.set_description("resize mrcs to width*height")
    for i in range(mrcs.shape[0]):
        mrc=mrcs[i]
        if is_norm:
            mrc = (mrc - np.min(mrc)) * (30 / (np.max(mrc) - np.min(mrc)))
            mrc = mrc - np.mean(mrc)
        mrc = PIL.Image.fromarray(mrc)
        resized_mrcs[i] = np.asarray(mrc.resize((width, height), PIL.Image.BICUBIC))
    resized_mrcs = resized_mrcs.astype('float32')
    return resized_mrcs

def norm(mrcs):
    norm_mrcs = np.zeros((mrcs.shape[0], mrcs.shape[1], mrcs.shape[2]))
    # pbar = tqdm(range(mrcs.shape[0]))
    # pbar.set_description("resize mrcs to width*height")
    for i in range(mrcs.shape[0]):
        mrc=mrcs[i]
        mrc = (mrc - np.min(mrc)) * (30 / (np.max(mrc) - np.min(mrc)))
        mrc = mrc - np.mean(mrc)
        norm_mrcs[i] = mrc
    norm_mrcs = norm_mrcs.astype('float32')
    return norm_mrcs

def get_resized_cropped_single_images(path_in, path_out, size, crop_ratio=None,is_norm=True):
    value_error_mrcs = []
    output_tifs_path=[]
    output_tifs_label=[]
    denoised_ouput_dirs_path=[]
    label=0

    if crop_ratio:
        mask = get_crop_mask(size, crop_ratio)

    if not os.path.exists(path_out+'/raw/'):
        os.makedirs(path_out+'/raw/')
    for root, dirs, files in os.walk(path_in):
        pbar = tqdm(files)
        pbar.set_description("mrcdata preprocessing")
        for  file in pbar:
            # out_dir = os.path.join(path_out, file)
            if os.path.splitext(file)[-1] == '.mrc' or os.path.splitext(file)[-1] == '.mrcs':
                try:
                    with mrcfile.open(path_in + file) as mrcdata:
                        mrcs = mrcdata.data
                        if mrcs.shape[1] > size:
                            resized_mrcs = mrcs_resize(mrcs, size, size,is_norm=is_norm)
                            is_norm=False
                        else:
                            resized_mrcs = mrcs
                        if is_norm:
                            resized_mrcs=norm(resized_mrcs)
                        if crop_ratio:
                            resized_mrcs=multi_process_crop(resized_mrcs,mask)
                        # if is_norm:
                        #     resized_mrcs=(resized_mrcs-np.min(resized_mrcs))*(30/(np.max(resized_mrcs)-np.min(resized_mrcs)))
                        #     resized_mrcs=resized_mrcs-np.mean(resized_mrcs)
                        mrcs_len = mrcs.shape[0]
                        if not os.path.exists(path_out+'/raw/'+file+'/'):
                            os.makedirs(path_out+'/raw/'+file+'/')
                        if not os.path.exists(path_out+'/denoised/'+file+'/'):
                            os.makedirs(path_out+'/denoised/'+file+'/')
                        for i in range(mrcs_len):
                            output_tif=path_out+'/raw/'+file+'/'+str(i)+'.tif'
                            denoised_output_dir=path_out+'/denoised/'+file+'/'+str(i)+'.tif'

                            PIL.Image.fromarray(resized_mrcs[i]).save(output_tif)
                            output_tifs_path.append(output_tif)
                            denoised_ouput_dirs_path.append(denoised_output_dir)
                            output_tifs_label.append(label)
                        label+=1

                        # projectons_file = mrcfile.new(
                        #     path_out + '/' + file,
                        #     resized_mrcs, overwrite=True)
                        # projectons_file.close()
                except ValueError:
                    print('ValueError of ' + file)
                    value_error_mrcs.append(file)
    if value_error_mrcs:
        filename = open(path_out + '/ValueError.txt', 'w')
        filename.write(str(value_error_mrcs))
        filename.close()
    # output_tifs_path=np.asarray(output_tifs_path)
    # denoised_ouput_dirs_path=np.asarray(denoised_ouput_dirs_path)
    with open(path_out+'output_tifs_path.data','wb') as filehandle:
        pickle.dump(output_tifs_path,filehandle)
    with open(path_out+'denoised_output_dirs_path.data','wb') as filehandle:
        pickle.dump(denoised_ouput_dirs_path,filehandle)
    with open(path_out+'output_tifs_label.data','wb') as filehandle:
        pickle.dump(output_tifs_label,filehandle)
    means_stds = get_mean_std(output_tifs_path)
    with open(path_out+'means_stds.data','wb') as filehandle:
        pickle.dump(means_stds,filehandle)

def get_mean_std(path_list,Cnum=10000):
    # calculate means and stds
    imgs=[]
    random.shuffle(path_list)
    if len(path_list)<Cnum:
        Cnum=len(path_list)
    for i in range(Cnum):
        img = np.array(Image.open(path_list[i]))
        imgs.append(img)
    imgs_np=np.asarray(imgs)
    means=imgs_np.mean()
    stds=imgs_np.std()
    return means,stds

def multi_process_crop(mrcsArray,mask,is_norm=False):
    # print('mrcs array is cropping:')
    items = [mrcsArray[i] for i in range(mrcsArray.shape[0])]
    pool = multiprocessing.Pool(8)
    # pbar = tqdm(items)
    # pbar.set_description("mrcs array cropping")
    func = partial(image_cropping, mask=mask)
    mrcsArray = pool.map(func, items)
    pool.close()
    pool.join()
    return np.asarray(mrcsArray)

def get_crop_mask(mrc_length, crop_ratio):
    mask = np.zeros([mrc_length, mrc_length])
    center = np.array([mrc_length / 2, mrc_length / 2])
    for i in range(mrc_length):
        for j in range(mrc_length):
            if math.sqrt(
                    (i - int(center[0])) ** 2 + (j - int(center[1])) ** 2) < crop_ratio*mrc_length*0.5:
                mask[i, j] = 1
    return mask


def image_cropping(img,mask):
    return img*mask


def get_single_images2(path_in, path_out, width, height):
    for root, dirs, files in os.walk(path_in):
        for index, file in enumerate(files):
            out_dir = os.path.join(path_out, str(index))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with mrcfile.open(path_in + file) as mrcdata:
                mrcs = mrcdata.data
                if mrcs.shape[1] > width:
                    resized_mrcs = mrcs_resize(mrcs, width, height)
                else:
                    resized_mrcs = mrcs
                mrcs_len = mrcs.shape[0]
                for i in range(mrcs_len):
                    projectons_file = mrcfile.new(
                        out_dir + '/' + str(i) + '.mrc',
                        resized_mrcs[i], overwrite=True)
                    projectons_file.close()
                # if not os.path.exists(path_out + '/'):
                #     os.makedirs(path_out + '/')
                # projectons_file = mrcfile.new(
                #     path_out + '/resized_' + file,
                #     resized_mrcs, overwrite=True)
                # projectons_file.close()
                # if index == 0:
                #     projections = projection
                # else:
                #     projections = np.append(projections, projection, axis=0)
            # label = [file.strip().split('_')[-1].split('.')[0]] * projection.shape[0]
            # labels = labels + label
    # print(type(projections), projections.shape, type(labels), len(labels))
    # return projections


if __name__ == '__main__':
    # path_in = '/mnt/data_synology/yanyang/data/em_dataset/real_particles/2021_virus_39000/P16_W27_particle/'
    # path_in = '/mnt/data_synology/yanyang/project/old/resnet_vae_clustering/result/2021_11_10_18_51_09/generated_class_single_particles/epoch0/'
    # path_in = '/mnt/data_synology/yanyang/data/em_dataset/real_particles/2021_virus_39000/P16_W27_particle/'
    # path_in = '/home/yanyang/data/em_dataset/real_particles/EMPIAR_10455/archive/10455/data/'
    path_in = '/home/yanyang/data/em_dataset/simulated_particles/from_scipion/emd_6840/snr_0.05/'
    # path_out = '/mnt/data_synology/yanyang/data/em_dataset/real_particles/2021_virus_39000/P16_W27_particle_single/'
    # path_out = '/mnt/data_synology/yanyang/data/em_dataset/real_particles/2021_virus_39000/P16_W27_particle_denoised_single/'
    path_out = '/home/yanyang/data/projects_resluts/scan_kmeans/2022_4_8_dataloader_test/preprocessed_data/'
    # path_out = '/mnt/data_synology/yanyang/data/em_dataset/real_particles/2021_virus_39000/P16_W27_particle_480_single/'
    if not os.path.exists(path_out+'/output_tifs_path.data'):
        get_resized_cropped_single_images(path_in, path_out, 128, 128)
    with open(path_out+'output_tifs_path.data','rb') as filehandle:
        output_tifs_path=pickle.load(filehandle)
    pass

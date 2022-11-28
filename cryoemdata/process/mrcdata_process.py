import mrcfile
# import tifffile
import os
import PIL.Image
import multiprocessing
import numpy as np
from tqdm import tqdm
import Running_Paras
import matplotlib.pyplot as plt
import math
from munkres import Munkres, print_matrix
from sklearn import metrics
from functools import partial
import torch
from EMAN2 import *
from torchvision.utils import save_image
import stat
import shutil
import cv2


# def get_nearest_img_to_center(mrcArry,centers,k):
#     imgs_nearest_to_centers=np.full(k,np.inf)
#     for j in range(k):
#         subcenter=centers[j]
#     return imgs_nearest_to_centers
def best_map(L1, L2):
    # L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def acc_clustering(gt_s, s):
    c_x = best_map(gt_s, s)
    count = np.sum(gt_s[:] == c_x[:])
    acc = count.astype(float) / (gt_s.shape[0])
    return acc


def performance_clustering(labels_real, labels_predict):
    label_same = best_map(labels_real, labels_predict)
    count = np.sum(labels_real[:] == label_same[:])
    acc = count.astype(float) / (labels_real.shape[0])
    nmi = metrics.normalized_mutual_info_score(labels_real, label_same)
    return acc, nmi


def get_coordinates(angles):
    nimg = angles.shape[0]
    coordinates = np.empty((nimg, 3))
    for i in range(nimg):
        psi = math.radians(angles[i, 0])
        theta = math.radians(angles[i, 1])
        x = math.sin(theta) * math.cos(psi)
        y = math.sin(theta) * math.sin(psi)
        z = math.cos(theta)
        coordinates[i] = np.array([x, y, z])
    return coordinates


def calc_angle_dist(angles, coordinates, i, j):
    inner_dot = np.dot(coordinates[i], coordinates[j])
    inner_dot = round(inner_dot, 8)
    tmp = math.degrees(math.acos(inner_dot))
    if tmp > 90:
        tmp = 180 - tmp
    print(angles[i], angles[j])
    print(coordinates[i], coordinates[j])
    print(tmp)


# input mrc files dir path, output mrc data ndarray.
def mrcs2num_array(path):
    classlist = []
    for root, dirs, _ in os.walk(path):
        # mrcdata = []
        # dirs=dirs.sort()
        i = 0
        for dir in dirs:
            for _, _, files in os.walk(root + dir + '/'):
                for file_name in files:
                    mrc = mrcfile.open(path + dir + '/' + file_name)
                    mrcdata = mrc.data
                    if mrcdata.ndim == 2:
                        mrcdata = np.expand_dims(mrcdata, axis=0)
                    if i == 0:
                        return_mrcdata = mrcdata
                        i = i + 1
                        classlist = classlist + [int(dir)]
                    else:
                        return_mrcdata = np.append(return_mrcdata, mrcdata, axis=0)
                        classlist = classlist + [int(dir)]
                    # print('load mrcfiles: ' + file_name)
                    mrc.close()
    return return_mrcdata, classlist


def loading_mrcs(path):
    labels = []
    for root, dirs, files in os.walk(path):
        files.sort()

        for index, file in enumerate(files):
            if os.path.splitext(file)[-1] == '.mrc' or os.path.splitext(file)[-1] == '.mrcs':
                with mrcfile.open(path + file) as mrcdata:
                    projection = mrcdata.data
                    if index == 0:
                        projections = projection
                    else:
                        projections = np.append(projections, projection, axis=0)
                # label = [file.strip().split('_')[-1].split('.')[0]] * projection.shape[0]
                label = [index] * projection.shape[0]
                labels = labels + label
    print(type(projections), projections.shape, type(labels), len(labels))
    return projections, np.array(labels)


# def loading_generated_projections_tiff(path):
#     labels = []
#     projections = []
#     for root, dirs, files in os.walk(path):
#         for index, file in enumerate(files):
#             projection = tifffile.imread(path + file)
#             print(projection.shape)
#             projections.append(projection)
#             label = [file.strip().split('_')[-1].split('.')[0]] * projection.shape[0]
#             labels = labels + label
#     projections = np.stack(projections)
#     print(type(projections), projections.shape, type(labels), len(labels))
#     return projections, labels


# input mrc data ndarray and number of particles, output flatten mrc data ndarray.
def flat(mrc, length):
    Y = []
    for i in range(length):
        X = mrc[i].flatten()
        Y.append(X)
    return np.array(Y)


def mrcs2mrc(mrcsdata):
    for i in range(mrcsdata.shape[0]):
        with mrcfile.new('./mrcdata/1/' + str(i) + '.mrc', overwrite=True) as mrc:
            mrc.set_data(mrcsdata[i])


def mrc_rotate(mrc, angles):
    mrc = PIL.Image.fromarray(mrc)
    mrc = mrc.rotate(360 - eval(angles))
    mrc = np.asarray(mrc)
    return mrc


def mrcs_resize(mrcs, width, height):
    resized_mrcs = np.zeros((mrcs.shape[0], width, height))
    pbar = tqdm(range(mrcs.shape[0]))
    pbar.set_description("resize mrcs to width*height")
    for i in pbar:
        mrc = PIL.Image.fromarray(mrcs[i])
        resized_mrcs[i] = np.asarray(mrc.resize((width, height), PIL.Image.BICUBIC))
    resized_mrcs = resized_mrcs.astype('float32')
    return resized_mrcs


def get_coordinates_of_angles(angles):
    nimg = angles.shape[0]
    coordinates = np.empty((nimg, 3))
    for i in range(nimg):
        psi = math.radians(angles[i, 0])
        theta = math.radians(angles[i, 1])
        x = math.sin(theta) * math.cos(psi)
        y = math.sin(theta) * math.sin(psi)
        z = math.cos(theta)
        coordinates[i] = np.array([x, y, z])
    return coordinates


def calc_angle_dist(angles, coordinates, i, j):
    inner_dot = np.dot(coordinates[i], coordinates[j])
    inner_dot = round(inner_dot, 8)
    tmp = math.degrees(math.acos(inner_dot))
    if tmp > 90:
        tmp = 180 - tmp
    print(angles[i], angles[j])
    print(coordinates[i], coordinates[j])
    print(tmp)


# mrc_add_gaussian_noise(Running_Paras.input_processed_mrc_path+'1/',Running_Paras.add_noise_mrc_path+'1/')
def get_averages(raw_images, labels, clustering_num):
    show_width = int(math.sqrt(clustering_num)) + 1
    plt.figure(figsize=(20, 20))
    # averages = raw_images[0]

    for i in range(clustering_num):
        # find children and generate averages
        children = raw_images[labels == i]
        average = np.expand_dims(np.mean(children, 0), axis=0)
        # save the averages
        if i == 0:
            averages = average
        else:
            averages = np.append(averages, average, axis=0)
        print('class' + str(i) + ' ' + str(children.shape[0]))
    return averages


def class_average_stdVar(raw_images, reliable_list, labels, clustering_num):
    show_width = int(math.sqrt(clustering_num)) + 1
    plt.figure(figsize=(20, 20))
    class_number_arry = []
    # averages = raw_images[0]

    for i in range(clustering_num):
        # find children and generate averages
        # print(np.asarray(reliable_list)[labels == i])
        children = raw_images[labels == i]
        children = children[np.asarray(reliable_list)[labels == i] == True]
        # label_children=reliable_list
        average = np.expand_dims(np.mean(children, 0), axis=0)
        if i == 0:
            averages = average
        else:
            averages = np.append(averages, average, axis=0)
        print('class' + str(i) + ' ' + str(children.shape[0]))
        # paint it
        class_number_arry = class_number_arry + [children.shape[0]]
    # plt.show()
    # print('Between-class variance '+str(np.var(class_number_arry)))
    stdVar = np.std(class_number_arry, ddof=1)
    print('Standard deviation between classes ' + str(stdVar))
    return averages, stdVar, class_number_arry


# preprocess('my_dataset/rotate_data/', 120)

def get_rotated_projections_from_templates(templates_file, ouput_dir, rotate_times, snr):
    templates = mrcfile.open(templates_file)
    templates = templates.data
    print('Get temlates {}'.format(templates.shape))
    rotate_angle = int(360 / rotate_times)
    # Method 1 : the var of noise was divided by the var of itself
    for index, template in enumerate(templates):
        projections = []
        for step in range(rotate_times):
            rotated_projection = mrc_rotate(template, str(step * rotate_angle))
            noise = np.random.normal(0, (rotated_projection.var() / snr) ** 0.5, rotated_projection.shape)
            noisy_projection = rotated_projection + noise
            projections.append(noisy_projection)
        projections_array = np.stack(projections).astype(np.float32)
        projectons_file = mrcfile.new(ouput_dir + 'snr_' + str(snr) + '_projections_' + str(index + 1) + '.mrc',
                                      projections_array, overwrite=True)
        projectons_file.close()
    # Method 2 : the var of noise was divided by the var of whole dataset
    # for index,template in enumerate(templates):
    #     projections= []
    #     for step in range(rotate_times):
    #         rotated_projection = mrc_rotate(template,step*rotate_angle)
    #         projections.append(rotated_projection)
    #     projections_array = np.stack(projections).astype(np.float32)
    #     noise = np.random.normal(0, (projections_array.var() / snr)**0.5, projections_array.shape)
    #     noisy_projection = (projections_array + noise).astype(np.float32)
    #     projectons_file = mrcfile.new(ouput_dir+'snr_'+str(snr)+'_projections_'+str(index)+'.mrc',noisy_projection,overwrite=True)
    #     projectons_file.close()


def get_rotated_resized_projections_from_templates(templates_file, ouput_dir, rotate_times, snr, resize):
    templates = mrcfile.open(templates_file)
    templates = templates.data
    print('Get temlates {}'.format(templates.shape))
    rotate_angle = int(360 / rotate_times)
    # Method 1 : the var of noise was divided by the var of itself
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    for index, template in enumerate(templates):
        projections = []
        for step in range(rotate_times):
            rotated_projection = mrc_rotate(template, str(step * rotate_angle))
            rotated_projection = PIL.Image.fromarray(rotated_projection)
            rotated_projection = np.asarray(
                rotated_projection.resize((resize, resize),
                                          PIL.Image.BICUBIC))
            noise = np.random.normal(0, (rotated_projection.var() / snr) ** 0.5, rotated_projection.shape)
            noisy_projection = rotated_projection + noise
            projections.append(noisy_projection)
        projections_array = np.stack(projections).astype(np.float32)
        projectons_file = mrcfile.new(ouput_dir + 'snr_' + str(snr) + '_projections_' + str(index + 1) + '.mrc',
                                      projections_array, overwrite=True)
        projectons_file.close()


def get_averages_with_rotate_correction_with_generated_imgs(generated_imgs, raw_imgs, reliable_list, labels,
                                                            pionts_nearest_to_centers,
                                                            is_simulated, clustering_num):
    rotate_times = Running_Paras.final_rotate_times

    # plt.figure(figsize=(20, 20))
    # averages = raw_images[0]
    rotate_angles = int(360 / rotate_times)
    phbar = tqdm(range(clustering_num))
    phbar.set_description("rotating correction")
    num_mrcs = generated_imgs.shape[0]
    for j in phbar:
        # find children and generate averages
        children = generated_imgs[labels == j]
        children = children[np.asarray(reliable_list)[labels == j] == True]
        num_children = children.shape[0]
        raw_imgs_children = raw_imgs[labels == j]
        raw_mrc = generated_imgs[int(pionts_nearest_to_centers[j])]
        t = 0
        flat_raw_mrc = raw_mrc.flatten().reshape(1, Running_Paras.all_mrc_pixels)

        for i in range(num_children):
            min_dist = np.inf
            for k in range(rotate_times):
                rotated_mrc = mrc_rotate(children[i], str(rotate_angles * k))
                flat_rotated_mrc = rotated_mrc.flatten().reshape(1, Running_Paras.all_mrc_pixels)

                dist = np.linalg.norm(flat_rotated_mrc - flat_raw_mrc)
                if dist < min_dist:
                    min_dist = dist
                    rotated_raw_img = mrc_rotate(raw_imgs_children[i], str(rotate_angles * k))
                    children[i] = rotated_mrc
                    raw_imgs_children[i] = rotated_raw_img
                    m = 0
                    for n in range(num_mrcs):
                        if labels[n] == j:
                            if t == m:
                                generated_imgs[n] = rotated_mrc

                                raw_imgs[n] = rotated_raw_img
                                break
                            m = m + 1
            t = t + 1
        average = np.expand_dims(np.mean(children, 0), axis=0)
        average_raw_imgs = np.expand_dims(np.mean(raw_imgs_children, 0), axis=0)
        # save the averages
        if j == 0:
            averages = average
            averages_raw_imgs = average_raw_imgs
        else:
            averages = np.append(averages, average, axis=0)
            averages_raw_imgs = np.append(averages_raw_imgs, average_raw_imgs, axis=0)

    return averages, averages_raw_imgs, raw_imgs


def aligin_one_img(im, ref):
    im = EMNumPy.numpy2em(im)
    # im.process_inplace("normalize.edgemean")
    # if im["nx"]!=nx or im["ny"]!=ny :
    # 	im=im.get_clip(Region(old_div(-(nx-im["nx"]),2),old_div(-(ny-im["ny"]),2),nx,ny))
    # im.write_image("result/seq.mrc",-1)
    # ima=im.align("translational",ref0,{"nozero":1,"maxshift":old_div(ref0["nx"],4.0)},"ccc",{})
    ima = im.align("rotate_translate_tree", ref)
    # ima = EMNumPy.em2numpy(ima)
    return ima


def img_normalize(imgs):
    return (imgs - imgs.mean()) / imgs.std()


def imgs_align(imgs, labels,
               k, pionts_nearest_to_centers, path_result_dir, iteration_times=4, align_batch=10000, is_norm=True,
               img_type='/raw/', true_labels=None, raw_imgs_path=None):
    # plt.figure(figsize=(20, 20))
    # averages = raw_images[0]
    # rotate_angles = int(360 / rotate_times)
    phbar = tqdm(range(k))
    phbar.set_description("rotating correction")
    aligned_imgs_path = path_result_dir + "/tmp/aligned_imgs/" + img_type
    average_imgs_path = path_result_dir + "/averages/"
    # aligined_imgs = imgs.copy()
    # num_imgs = imgs.shape[0]
    # for j in range(k):
    total_num = 0
    if true_labels is not None:
        true_labels = np.array(true_labels)
    labels_after_align = []
    path_raw_to_align = []
    output_tifs_path = []
    if os.path.exists(aligned_imgs_path):
        delete_file(aligned_imgs_path)

    for j in phbar:
        # find children and generate averages
        if isinstance(imgs[0], str):
            children_path = imgs[labels == j]
            path_raw_to_align += children_path.tolist()
            # if raw_imgs_path is not None:
            #     children_raw_path=raw_imgs_path[labels==j]
            #     a=dict(zip(children_raw_path.tolist(),children_path.tolist()))
            #     a=dict(zip(children_raw_path,children_path))
            #     path_raw_to_align=dict(path_raw_to_align,**dict(zip(children_raw_path,children_path)))
            if true_labels is not None:
                labels_after_align += true_labels[labels == j].tolist()
            children_iter_times = int(children_path.shape[0] / align_batch)
            if children_path.shape[0] % align_batch > 0:
                children_iter_times += 1
            # children_path=children_path[0:10000] if children_path.shape[0]>10000 else children_path
            children = read_imgs_from_path_list(children_path)
            ref = EMNumPy.numpy2em(np.asarray(PIL.Image.open(imgs[int(pionts_nearest_to_centers[j])])))
            pass
        else:
            children = imgs[labels == j]
            children = children[0:10000] if children.shape[0] > 10000 else children
            children_iter_times = int(children.shape[0] / align_batch)
            if children.shape[0] % align_batch > 0:
                children_iter_times += 1
            ref = EMNumPy.numpy2em(imgs[int(pionts_nearest_to_centers[j])])
        # children = children[np.asarray(reliable_list)[labels == j] == True]
        num_children = children.shape[0]
        if true_labels is None:
            labels_after_align += [j] * num_children
        # raw_imgs_children = raw_imgs[labels == j]

        for i in range(iteration_times):
            # print("Iter ", i)
            avgr = Averagers.get("mean", {"ignore0": True})
            average_list_i = []
            aligned_list = []
            for ii in range(children_iter_times):
                # e=(ii+1)*align_batch if (ii+1)*align_batch<num_children else num_children+1
                if (ii + 1) * align_batch < num_children:
                    item = [children[ii * align_batch:(ii + 1) * align_batch][i] for i in range(align_batch)]
                else:
                    item = [children[ii * align_batch:][i] for i in range(num_children % align_batch)]
                # item = [children for i in range(num_children)]
                func = partial(aligin_one_img, ref=ref)
                pool = multiprocessing.Pool(10)
                aligined_img = pool.map(func, item)
                aligned_list.append(aligined_img)
                pool.close()
                pool.join()
                for num in range(len(aligined_img)):
                    avgr.add_image(aligined_img[num])
                    # aligined_imgs[total_num + num] = EMNumPy.em2numpy(aligined_img[num])

                    output_tif = aligned_imgs_path + '/' + str(j) + '/'
                    if not os.path.exists(output_tif):
                        os.makedirs(output_tif)
                    if i == iteration_times - 1:
                        img_i = EMNumPy.em2numpy(aligined_img[num])
                        if is_norm:
                            img_i = (img_i - np.min(img_i)) * 30 / (np.max(img_i) - np.min(img_i))
                            img_i = img_i - np.mean(img_i)
                        PIL.Image.fromarray(img_i).save(output_tif + str(ii * align_batch + num) + '.tif')
                        output_tifs_path.append(output_tif + str(ii * align_batch + num) + '.tif')

                ref = avgr.finish()
        average = EMNumPy.em2numpy(ref)
        average = np.expand_dims(average, axis=0)
        # save the averages
        if j == 0:
            averages = average.copy()
        else:
            averages = np.append(averages, average, axis=0)
        total_num += num_children
    if not os.path.exists(average_imgs_path):
        os.makedirs(average_imgs_path)
    # save_image(torch.unsqueeze(torch.from_numpy(averages), 1),
    #            average_imgs_path + '/clustering_result' + str(epoch) + '.png')
    raw_to_align_dict = dict(zip(path_raw_to_align, output_tifs_path))
    return averages, output_tifs_path, np.array(labels_after_align), raw_to_align_dict


def get_circle_mask(mrc_length, features_start, features_end):
    mask = np.zeros([mrc_length, mrc_length])
    for i in range(mrc_length):
        for j in range(mrc_length):
            if features_start < math.sqrt(
                    (i - int(mrc_length / 2)) ** 2 + (j - int(mrc_length / 2)) ** 2) < features_end:
                mask[i, j] = 1
    return mask


def get_masks(w, features_num, features_length):
    masks = []
    if features_num * features_length > w:
        features_num = int(w / features_length)
    for i in range(features_num):
        masks.append(get_circle_mask(w, i * features_length, (i + 1) * features_length))
    return masks


def get_mrc_circle_features(mrc, masks):
    # [w, h] = mrc.shape

    for i in range(len(masks)):
        # mask = get_circle_mask(w, i * features_length, (i + 1) * features_length)
        mask = masks[i]
        masked_mrc = mrc * mask
        feature = np.expand_dims(sum(sum(masked_mrc)), axis=0)
        if i == 0:
            circle_features = feature
        else:
            circle_features = np.append(circle_features, feature, axis=0)
    return circle_features


def get_mrc_features(mrc, type='LBP'):
    [w, h] = mrc.shape
    '''to uint8'''
    # mrc = (mrc - np.min(mrc)) * 255 / (np.max(mrc) - np.min(mrc))
    # # img2*=1/ (np.max(img) - np.min(img))
    # mrc = mrc.astype(np.uint8)
    if type == 'SIFT':
        '''sift'''
        mrc = mrc.astype(np.uint8)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(mrc, None)
        # reshape_feature = des[0].reshape(-1, 1)
        reshape_feature = np.mean(des, axis=0)
        feature = np.squeeze(reshape_feature)

        # s=reshape_feature[0:128]

    if type == 'HU':
        mrc = (mrc - np.min(mrc)) * 255 / (np.max(mrc) - np.min(mrc))
        '''hu moments'''
        moments = cv2.moments(mrc)
        moments_array = [v for k, v in moments.items()]
        humoments = cv2.HuMoments(moments)
        feature =np.squeeze( humoments)

    '''Fourier Descriptors'''
    #
    # from pyefd import elliptic_fourier_descriptors
    # contours, hierarchy = cv2.findContours(
    #     mrc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    '''test'''
    # cv2.drawContours(mrc, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("img", mrc)
    # cv2.waitKey(0)
    # # Iterate through all contours found and store each contour's
    # # elliptical Fourier descriptor's coefficients.
    # coeffs = []
    # for cnt in contours:
    #     # Find the coefficients of all contours
    #     coeffs.append(elliptic_fourier_descriptors(
    #         np.squeeze(cnt), order=10))
    # return np.squeeze(humoments)
    '''LBP'''
    if type == 'LBP':
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(mrc, 8, 2, 'uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        # return np.array(moments_array)
        feature = hist
    return feature


def get_mrcs_circle_features(mrcs, masks):
    item = [mrcs[i] for i in range(mrcs.shape[0])]
    # phbar = tqdm(item)
    '''multi thread'''
    # phbar.set_description("get circle features")
    pool = multiprocessing.Pool(Running_Paras.num_threads)
    func = partial(get_mrc_circle_features, masks=masks)
    circle_features = pool.map(func, item)
    pool.close()
    pool.join()
    '''single thread'''
    # circle_features=[]
    # for mrc in item:
    #     circle_features.append(get_mrc_circle_features(np.squeeze(mrc),masks))

    return np.asarray(circle_features)


def get_mrcs_other_features(mrcs, type='LBP'):
    item = [mrcs[i] for i in range(mrcs.shape[0])]
    # phbar = tqdm(item)
    # matcher = cv2.BFMatcher()
    '''multi thread'''
    # phbar.set_description("get circle features")
    # pool = multiprocessing.Pool(Running_Paras.num_threads)
    # sift_features = pool.map(get_mrc_sift_features, item)
    # pool.close()
    # pool.join()
    '''single thread'''
    features = []
    for mrc in item:
        features.append(get_mrc_features(np.squeeze(mrc), type=type))

    return np.asarray(features)


def get_mrcs_circle_features_from_path(val_loader, net, features_num, features_length):
    net.eval()
    feature_bank = []
    label_bank = []
    masks = get_masks(Running_Paras.resized_mrc_width, features_num, features_length)
    # '''save denoised images'''
    # with open(Running_Paras.path_result_dir+ '/preprocessed_data/denoised_output_dirs_path.data', 'rb') as filehandle:
    #     path_data = sorted(pickle.load(filehandle))

    '''extract features from denoised imgs'''
    with torch.no_grad():
        for data, _, label, _ in tqdm(val_loader, desc='Feature extracting'):
            denoised, _, _ = net.forward(data.cuda())
            feature_bank.extend(get_mrcs_circle_features(np.squeeze(denoised.cpu().numpy()), masks))
            label_bank.extend(label.cpu().numpy())
        # feature_bank = torch.cat(feature_bank, dim=0)
        # feature_bank = feature_bank.cpu().numpy()
    return np.asarray(feature_bank), np.asarray(label_bank)


def save_denoised_imgs(val_loader, net, path_result_dir, normalize=False, path_data=None, save_label=""):
    i_sum = 0
    label_bank = []
    path_bank = []
    net.eval()
    # val_loader.shuffle=False
    if path_data is None:
        with open(path_result_dir + '/tmp/preprocessed_data/denoised_output_dirs_path.data', 'rb') as filehandle:
            path_data = pickle.load(filehandle)
            # path_data = pickle.load(filehandle)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='saving denoised imgs'):
            # for data, _, label, _ in tqdm(val_loader, desc='saving denoised imgs'):
            denoised_tensor, _, _ = net.forward(batch['mrcdata'].cuda())
            denoised = np.squeeze(denoised_tensor.cpu().numpy())
            # data_np=np.squeeze(batch['mrcdata'].cpu().numpy())
            path_num = denoised.shape[0]
            label_bank.extend(batch['label'].cpu().numpy())
            path_bank += batch['path']
            for i in range(path_num):
                if normalize:
                    denoised[i] = (denoised[i] - np.min(denoised[i])) / (np.max(denoised[i]) - np.min(denoised[i]))
                PIL.Image.fromarray(denoised[i]).save(path_data[i + i_sum])
            i_sum += path_num
    label_bank = np.asarray(label_bank)
    save_image(torch.tensor(denoised_tensor),
               path_result_dir + '/tmp/preprocessed_data/denoised_img' + save_label)
    with open(path_result_dir + '/tmp/preprocessed_data/output_denoised_tifs_path.data', 'wb') as filehandle:
        pickle.dump(path_data, filehandle)
    with open(path_result_dir + '/tmp/preprocessed_data/output_denoised_tifs_label_path.data', 'wb') as filehandle:
        pickle.dump(label_bank, filehandle)
    return path_data, label_bank


def get_mrcs_circle_features_from_denoised_path(denoised_loader, features_num, features_length,
                                                using_spectrum_feature=False):
    feature_bank = []

    masks = get_masks(Running_Paras.resized_mrc_width, features_num, features_length)
    # '''save denoised images'''
    # with open(Running_Paras.path_result_dir+ '/preprocessed_data/denoised_output_dirs_path.data', 'rb') as filehandle:
    #     path_data = sorted(pickle.load(filehandle))

    '''extract features from denoised imgs'''
    with torch.no_grad():
        for denoised in tqdm(denoised_loader, desc='Feature extracting'):

            if using_spectrum_feature:
                F_denoised = imgs_Fourier_trans(denoised)
                feature_bank.extend(get_mrcs_circle_features(np.squeeze(F_denoised.cpu().numpy()), masks))
            else:
                feature_bank.extend(get_mrcs_circle_features(np.squeeze(denoised.cpu().numpy()), masks))

        # feature_bank = torch.cat(feature_bank, dim=0)
        # feature_bank = feature_bank.cpu().numpy()
    return np.asarray(feature_bank)


def imgs_Fourier_trans(imgs):
    item = [imgs[i] for i in range(imgs.shape[0])]
    tranformed_imgs = []
    # phbar = tqdm(item)
    # matcher = cv2.BFMatcher()
    '''multi thread'''
    # phbar.set_description("get circle features")
    # pool = multiprocessing.Pool(Running_Paras.num_threads)
    # sift_features = pool.map(get_mrc_sift_features, item)
    # pool.close()
    # pool.join()
    '''single thread'''
    # sift_features = []
    for img in item:
        fft2_image = np.fft.fft2(img)
        shift2center = np.fft.fftshift(fft2_image)
        # tranformed_imgs.append(np.log(np.abs(shift2center)))
        tranformed_imgs.append(np.abs(shift2center))
    return np.asarray(tranformed_imgs)


def get_mrcs_other_features_from_denoised_path(denoised_loader, features_type='LBP'):
    feature_bank = []

    # '''save denoised images'''
    # with open(Running_Paras.path_result_dir+ '/preprocessed_data/denoised_output_dirs_path.data', 'rb') as filehandle:
    #     path_data = sorted(pickle.load(filehandle))

    '''extract features from denoised imgs'''
    with torch.no_grad():
        for denoised in tqdm(denoised_loader, desc='Feature extracting'):
            feature_bank.extend(get_mrcs_other_features(denoised.numpy(), type=features_type))

        # feature_bank = torch.cat(feature_bank, dim=0)
        # feature_bank = feature_bank.cpu().numpy()
    return np.asarray(feature_bank)


def read_imgs_from_path_list(paths):
    imgs = []
    for path in paths:
        imgs.append(np.asarray(PIL.Image.open(path)))
    return np.asarray(imgs)


# filePath:文件夹路径
def delete_file(filePath):
    if os.path.exists(filePath):
        for fileList in os.walk(filePath):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(filePath)
        return "delete ok"
    else:
        return "no filepath"

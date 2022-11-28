from EMAN2 import *
from tqdm import tqdm
from functools import partial
import numpy as np
import multiprocessing


def aligin_one_img(im, ref):
    im = EMNumPy.numpy2em(im)
    # im.process_inplace("normalize.edgemean")
    # if im["nx"]!=nx or im["ny"]!=ny :
    # 	im=im.get_clip(Region(old_div(-(nx-im["nx"]),2),old_div(-(ny-im["ny"]),2),nx,ny))
    # im.write_image("result/seq.mrc",-1)
    # ima=im.align("translational",ref0,{"nozero":1,"maxshift":old_div(ref0["nx"],4.0)},"ccc",{})
    ima = im.align("rotate_translate_tree", ref)
    # ima.write_image("seq.mrc", -1)
    # print(fsp, ima["xform.align2d"], ima.cmp("ccc", ref))
    ima.process_inplace("normalize.toimage", {"to": ref, "ignore_zero": 1})
    return ima


def imgs_align(imgs, labels,
               k, pionts_nearest_to_centers):
    # plt.figure(figsize=(20, 20))
    # averages = raw_images[0]
    # rotate_angles = int(360 / rotate_times)
    phbar = tqdm(range(k))
    phbar.set_description("rotating correction")
    aligined_imgs = imgs.copy()
    # num_imgs = imgs.shape[0]
    # for j in range(k):
    total_num = 0
    for j in phbar:
        # find children and generate averages
        children = imgs[labels == j]
        # children = children[np.asarray(reliable_list)[labels == j] == True]
        num_children = children.shape[0]
        # raw_imgs_children = raw_imgs[labels == j]
        ref = EMNumPy.numpy2em(imgs[int(pionts_nearest_to_centers[j])])
        for i in range(2):
            # print("Iter ", i)
            avgr = Averagers.get("mean", {"ignore0": True})
            item = [children[i] for i in range(num_children)]
            func = partial(aligin_one_img, ref=ref)
            pool = multiprocessing.Pool(10)
            aligined_img = pool.map(func, item)
            pool.close()
            pool.join()
            for num in range(len(aligined_img)):
                avgr.add_image(aligined_img[num])
                aligined_imgs[total_num + num] = EMNumPy.em2numpy(aligined_img[num])
            ref = avgr.finish()
        average = EMNumPy.em2numpy(ref)
        average = np.expand_dims(average, axis=0)
        # save the averages
        if j == 0:
            averages = average
        else:
            averages = np.append(averages, average, axis=0)
        total_num += num_children
    # if not os.path.exists(save_path + 'averages/'):
    #     os.makedirs(save_path + 'averages/')
    # save_image(torch.unsqueeze(torch.from_numpy(averages), 1),
    #            save_path + 'averages/clustering_result_' + str(epoch) + '.png')
    return averages, aligined_imgs

def align_test(path_in,path_out):
    # find children and generate averages
    children = imgs[labels == j]
    # children = children[np.asarray(reliable_list)[labels == j] == True]
    num_children = children.shape[0]
    # raw_imgs_children = raw_imgs[labels == j]
    ref = EMNumPy.numpy2em(imgs[int(pionts_nearest_to_centers[j])])
    for i in range(2):
        # print("Iter ", i)
        avgr = Averagers.get("mean", {"ignore0": True})
        item = [children[i] for i in range(num_children)]
        func = partial(aligin_one_img, ref=ref)
        pool = multiprocessing.Pool(10)
        aligined_img = pool.map(func, item)
        pool.close()
        pool.join()
        for num in range(len(aligined_img)):
            avgr.add_image(aligined_img[num])
            aligined_imgs[total_num + num] = EMNumPy.em2numpy(aligined_img[num])
        ref = avgr.finish()
    average = EMNumPy.em2numpy(ref)
    average = np.expand_dims(average, axis=0)
    # save the averages
    if j == 0:
        averages = average
    else:
        averages = np.append(averages, average, axis=0)
    total_num += num_children
def vis_test():
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_iris, load_digits
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import os

    digits = load_digits()
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(digits.data)
    X_pca = PCA(n_components=2).fit_transform(digits.data)

    ckpt_dir = "images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, label="t-SNE")
    plt.legend()
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, label="PCA")
    plt.legend()
    plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()
if __name__ == "__main__":
    path_in = ''
    path_out = ''
    vis_test()



# @Auther : Written/modified by Yang Yan
# @Time   : 2021/12/15 上午11:09
# @E-mail : yanyang98@yeah.net
# @Function : Selecting classes and generate star file.
import numpy as np


def selected_mrcs_star_file(original_star_file, predict_labels, save_path,selected_class):
    class_id=set(selected_class)

    with open(
            original_star_file,
            "r") as starfile:  # 打开文件
        data = starfile.readlines()  # 读取文件
    for index,x in enumerate(data):

        if x=='data_particles\n':
            for index2,x2 in enumerate(data[index:]):

                splited_x = x2.split()
                next_splited_x = data[index+index2 + 1].split()
                if splited_x:
                    item_num=splited_x[-1].replace("#", "")
                    if item_num.isdigit():
                        if int(item_num)==len(next_splited_x) and int(item_num)!=len(splited_x):
                            start_site=index+index2+1
                            break

    predict_labels = np.load(predict_labels)
    # for i in range(predict_labels):
    #     if i not in class_id:
    for index,x in enumerate(data[::-1]):
        # print(len(x.strip()))
        if x.strip():
            break
        else:
            data.pop()
    number_for_each_class=[]

    for selected in class_id:
        number_for_each_class+=predict_labels[predict_labels==selected].shape
    kkk = sum(number_for_each_class)
    selected_star_list=[x for index,x in enumerate(data[start_site:]) if len(x)>0 and predict_labels[index] in class_id ]
    print(str(len(selected_star_list))+' particles are selected.')
    new_star_list=data[:start_site]+selected_star_list
    file = open(save_path+'/selected_particles'+str(len(selected_star_list))+'.star', 'w')
    file.write(''.join(new_star_list))
    print('Successfully generated star file!')
    print('number for each selected class: '+str(number_for_each_class))
    print('total number: '+str(kkk))


# my_star_file=
# selected_classes=[0,3,4,8,13,19,20,28,36,47,43,1,11,12,16,21,26,27,29,31,32,35,37,38,49,50,51,52,53,55,56,58,61,63]# for epoch 204
# selected_classes=[0,1,3,4,5,6,7,11,12,14,15,16,17,21,22,23,25,26,27,28,29,31]# for epoch 200
# selected_classes=[0,1,3,4,5,11,12,14,15,16,17,21,22,23,25,26,27,28,29,31]# for epoch 200
# selected_classes=[0,1,2,3,4,7,8,9,10,15,16,17,19,22,23,25,26,27,30]# for epoch 199
# selected_classes=[1,2,4,5,6,7,8,15,16,21,22,23,24,26,27,30]# for epoch 200_v2
selected_classes=[1,2,8,10,16,17,19,28,29,30,31,32,33,35,39,43,44,45,46,49]# for resnet vae epoch 0
# selected_classes=[14,42]# for resnet vae epoch 0
# selected_classes=[1,2,4,5,6,7,8,9,11,13,14,17,18,21,22,23,24,25,26,28,29,31,32,34,36,37,41,42,43,44,49]# t20s select
# selected_classes=[0,2,3,5,6,7,8,11,14,15,18,19,20,21,23,24,25,26,27,31,32,34,35,36,37,38,40,42,43,44,45,48,47,46]# t20s select

# selected_mrcs_star_file(
#     "/home/yanyang/data/em_dataset/real_particles/galactosidase/from_relion_extract/003784_ProtRelionExportParticles/Export/particles_003784.star",
#     '/home/yanyang/data/projects_resluts/scan_kmeans/2021_12_14_galactosidase_test_results/cryoEM/pretext/clustering_labels/predict_labels_epoch200.npy',
# '/home/yanyang/data/em_dataset/real_particles/galactosidase/from_relion_extract/003784_ProtRelionExportParticles/Export/',selected_classes)
# selected_mrcs_star_file(
#     "/home/yanyang/data/relion_data/tmp/particles.star",
#     '/home/yanyang/data/projects_resluts/scan_kmeans/2021_12_21_galactosidase_test_results/cryoEM/pretext/clustering_labels/predict_labels_epoch200.npy',
# '/home/yanyang/data/relion_data/tmp/',selected_classes)

# selected_mrcs_star_file(
#     "/home/yanyang/data/relion_data/tmp/particles.star",
#     '/home/yanyang/data/project/resnet_vae_clustering/result/galactosidase_test_2021_12_22/clustering_labels/predict_labels_epoch0.npy',
# '/home/yanyang/data/relion_data/tmp/',selected_classes)

selected_mrcs_star_file(
    "/home/yanyang/data/relion_data/tmp/t20s/from_P2_J170.star",
    '/home/yanyang/data/project/resnet_vae_clustering/result/formal_t20s_2022_1_20_test1/clustering_labels/predict_labels_epoch0.npy',
'/home/yanyang/data/relion_data/tmp/t20s/',selected_classes)
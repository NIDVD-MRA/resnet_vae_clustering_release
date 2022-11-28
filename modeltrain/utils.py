# @Author : Written/modified by Yang Yan
# @Time   : 2022/6/20 下午7:08
# @E-mail : yanyang98@yeah.net
# @Function :
import os
import numpy as np
from torchvision.utils import save_image
import Running_Paras

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_clustering_labels(labels_path, epoch, clustering_labels):
    if not os.path.exists(labels_path):
        os.makedirs(labels_path + '/')
    np.save(labels_path + '/predict_labels_epoch' + str(epoch) + '.npy', clustering_labels)
def save_acc_data(acc, acc_best, nmi, nmi_best, tb_writer, acc_sum, nmi_sum, clustering_times, epoch, out_path):
    if acc > acc_best:
        acc_best = acc
    if nmi > nmi_best:
        nmi_best = nmi
    acc_sum = acc_sum + acc
    nmi_sum = nmi + nmi_sum
    clustering_times = clustering_times + 1
    tb_writer.add_scalar("accuracy:", acc, epoch)
    tb_writer.add_scalar("nmi:", nmi, epoch)
    tb_writer.add_scalar("mean accuracy:", acc_sum / clustering_times,
                         clustering_times)
    tb_writer.add_scalar("mean NMI:", nmi_sum / clustering_times,
                         clustering_times)
    if not os.path.exists(out_path + 'acc_data/'):
        os.makedirs(out_path + 'acc_data/')
    with open(out_path + 'acc_data/' + 'acc_data.txt', 'w') as average_acc:
        average_acc.write('best accuracy' + str(acc_best))
        average_acc.write('\nbest NMI' + str(nmi_best))
        average_acc.write('\naverage accuracy' + str(acc_sum / clustering_times))
        average_acc.write('\naverage NMI' + str(nmi_sum / clustering_times))
    return acc_best, nmi_best, acc_sum, nmi_sum, clustering_times

def save_trained_model(model, optimizer, epoch, save_path):
    import torch
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                'epoch': epoch + 1}, save_path + 'epoch' + str(epoch) + '_model.pth.tar')

def save_averages(average_imgs_all, average_generated_imgs=None, run_root_dir=None, epoch=0):
    import torch
    import mrcfile
    if not os.path.exists(run_root_dir + 'averages/'):
        os.makedirs(run_root_dir + 'averages/')
    save_image(torch.unsqueeze(torch.from_numpy(average_imgs_all), 1),
               run_root_dir + 'averages/clustering_result_' + str(epoch) + '.png')
    if average_generated_imgs is not None:
        save_image(torch.unsqueeze(torch.from_numpy(average_generated_imgs), 1),
                   run_root_dir + 'averages/generated_clustering_result_' + str(epoch) + '.png')
        # projectons_file = mrcfile.new(
        #     run_root_dir + 'averages/generated_clustering_averages_' + str(epoch) + '.mrcs',
        #     average_generated_imgs, overwrite=True)
    projectons_file = mrcfile.new(
        run_root_dir + 'averages/clustering_averages_' + str(epoch) + '.mrcs',
        average_imgs_all, overwrite=True)
    projectons_file.close()

def save_imgs_one_epoch(model, old_loader, epoch, run_root_dir):

    import torch
    model.eval()
    one_batch=next(iter(old_loader))
    img = one_batch['mrcdata'].cuda()
    with torch.no_grad():
        gen_img, mu_img, logvar_img = model.forward(img)
    img_contrast = torch.cat([img, gen_img], dim=0)
    if not os.path.exists(run_root_dir + 'AE_iterations/'):
        os.makedirs(run_root_dir + 'AE_iterations/')
        # Show the unre_imgs of last batch and its generated imgs

    save_image(to_img(img_contrast),
               run_root_dir + 'AE_iterations/' + 'round1_epoch_{}_img_gen_old_model.png'.format(
                   epoch + 1))

def to_img(x):
    x = x.view(x.size(0), 1, Running_Paras.resized_mrc_width, Running_Paras.resized_mrc_height)
    return x

def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# def save_classified_particles(run_root_dir, epoch, generated_imgs, clustering_labels, class_number_arry,cluster_num):
#     if not os.path.exists(run_root_dir + 'class_single_particles/epoch' + str(epoch) + '/'):
#         os.makedirs(run_root_dir + 'class_single_particles/epoch' + str(epoch) + '/')
#     if not os.path.exists(run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/'):
#         os.makedirs(run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/')
#     if Running_Paras.is_save_clustered_single_particles:
#         for i in range(cluster_num):
#             # class_single_particles = raw_mrcArrays[clustering_labels == i]
#             generated_class_single_particles = generated_imgs[clustering_labels == i]
#             generated_class_single_particles = np.squeeze(generated_class_single_particles)
#             class_particles = mrcfile.new(
#                 run_root_dir + 'generated_class_single_particles/epoch' + str(epoch) + '/' + 'class_' + str(
#                     i) + "_" + str(class_number_arry[i]) + '.mrcs',
#                 generated_class_single_particles, overwrite=True)
#             class_particles.close()
#         evaluate_utils.classify_mrcs(clustering_labels, cluster_num,
#                                      Running_Paras.path_for_data_classify,
#                                      run_root_dir + 'class_single_particles/epoch' + str(epoch) + '/')
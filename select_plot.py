import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, general_knn_monitor, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from datetime import datetime
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from typing import Tuple
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, kmeans_plusplus


def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if classifier is not None:
                outputs = classifier(outputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


def main(device, args):

    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(
        args)

    result_path = os.path.join("./figures", args.dataset_kwargs['dataset'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    scl_prefix = '_scl' if args.cl_default else ''
    result_plot_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}{args.utils.comment}_eval.png'
    result_sep_plot_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}{args.utils.comment}_sep_eval.png'
    # result_recording_path = result_path + '/' + \
    #     args.model.cl_model + scl_prefix + '_eval.txt'
    # result_plot_path = result_path + '/' + args.model.cl_model + scl_prefix + '_eval.png'
    # result_sep_plot_path = result_path + '/' + args.model.cl_model + scl_prefix + '_sep_eval.png'

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
    model = get_model(args, device, len(train_loader),
                      dataset.get_transform(args))

    print("model device:", model.device)

    acc_matrix = np.zeros((dataset.N_TASKS, dataset.N_TASKS))
    avg_acc = []
    fwt = []
    bwt = []

    if args.method != 'joint':
        for t in range(dataset.N_TASKS):
            _, _, _ = dataset.get_data_loaders(
                args)
            
            if t != 3:
                continue            

            model_path = os.path.join(
                args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}{args.utils.comment}_{t}.pth")
            print(f'loading from {model_path}')
            # model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
            model.cpu()
            model.net.load_state_dict(torch.load(model_path)['state_dict'])
            # model.net.backbone.load_state_dict({k[9:]:v for k, v in torch.load(model_path)['state_dict'].items() if 'backbone.' in k}, strict=True)
            model.to(device)
            model.eval()

            feature_bank = []
            image_bank = []
            # aug_image_bank = []
            label_bank = []
            with torch.no_grad():
                # generate feature bank
                for idx, ((_, images2, notaug_images), labels) in enumerate(dataset.train_loaders[-1]):
                # for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=True):
                    feature = model.net.encoder(notaug_images.to(device))
                    # feature = F.normalize(feature, dim=1)
                    feature_bank.append(feature)
                    image_bank.append(notaug_images)
                    # aug_image_bank.append(images2)
                    label_bank += labels
                # [D, N]
                feature_bank = torch.cat(feature_bank, dim=0).contiguous()
                image_bank = torch.cat(image_bank, dim=0).contiguous()
                # aug_image_bank = torch.cat(aug_image_bank, dim=0).contiguous()
                label_bank = np.array(label_bank)

                feature_bank = F.normalize(feature_bank, dim=1)
                print(feature_bank.shape)
                # print(feature_bank[0, :].view(1, -1).shape)

            if args.train.cluster_type == 'k-means':
                clustering = KMeans(n_clusters=args.model.buffer_size).fit(feature_bank.cpu())
                centers = clustering.cluster_centers_
                cluster_labels = clustering.labels_
                indices = pairwise_distances_argmin(centers, feature_bank.cpu())
                
                std_list = []
                for i in range(args.model.buffer_size):
                    mask = cluster_labels == i
                    cluster_features = feature_bank[mask, :]
                    std = torch.std(cluster_features, dim=0)
                    std_list.append(torch.mean(std))
            
            elif args.train.cluster_type == 'hierarchical':
                clustering = AgglomerativeClustering(n_clusters=args.model.buffer_size, linkage=args.train.linkage).fit(feature_bank.cpu())
                cluster_labels = clustering.labels_
                centers = []
                std_list = []
                for i in range(args.model.buffer_size):
                    mask = cluster_labels == i
                    cluster_features = feature_bank[mask, :]
                    center = torch.mean(cluster_features.cpu(), dim=0)
                    # center = np.mean(cluster_features, axis=0)
                    centers.append(center)
                    std = torch.std(cluster_features, dim=0)
                    std_list.append(torch.mean(std))
                centers = torch.stack(centers).contiguous()
                indices = pairwise_distances_argmin(centers, feature_bank.cpu())

            elif args.train.cluster_type == 'uniform':
                if args.train.remove_outliers:
                    print("removing outliers!")
                    feature_bank = feature_bank.cpu().numpy()
                    neigh = NearestNeighbors(n_neighbors=10)
                    neigh.fit(feature_bank)
                    neigh_dist = neigh.kneighbors(feature_bank)[0]
                    # print(neigh_dist.shape)
                    mean_neigh_dist = np.mean(neigh_dist, axis=1)
                    # print(mean_neigh_dist.shape)
                    core_indices = np.argsort(mean_neigh_dist)[:-100]
                    # print(core_indices.shape)
                    centers, indices = kmeans_plusplus(feature_bank[core_indices], n_clusters=args.model.buffer_size)
                else:
                    feature_bank = feature_bank.cpu().numpy()
                    centers, indices = kmeans_plusplus(feature_bank, n_clusters=args.model.buffer_size)
                std_list = None             

            
            feature_embedded = TSNE(n_components=2, learning_rate='auto', perplexity=50,
            init='pca').fit_transform(feature_bank)
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
            
            # label_color = np.array([colors[i] for i in label_bank])

            for i, label in enumerate(list(set(label_bank))):
                mask = label_bank == label
                ax.scatter(feature_embedded[mask, 0], feature_embedded[mask, 1], c=colors[i], label=f"class {i}")
            
            ax.scatter(feature_embedded[indices, 0], feature_embedded[indices, 1], c='black', s=120, label='selected')

            if t == 3:
                break

             


    # plt.show()
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_plot_path)

        # # tot_fig.tight_layout()
        # sep_fig.tight_layout()
        # # tot_fig.savefig(result_plot_tot_path)
        # sep_fig.savefig(result_sep_plot_path)
        # fig.tight_layout()
        # fig.savefig(result_plot_path)



if __name__ == "__main__":
    for i in range(1):
        args = get_args()
        print("device is:", args.device)

        main(device=args.device, args=args)
        # completed_log_dir = args.log_dir.replace(
        #     'in-progress', 'debug' if args.debug else 'completed')
        # os.rename(args.log_dir, completed_log_dir)
        # print(f'Log file has been saved to {completed_log_dir}')

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

    result_path = os.path.join("./results", args.dataset_kwargs['dataset'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    scl_prefix = '_scl' if args.cl_default else ''
    result_recording_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}_linear_eval.txt'
    # result_plot_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}_eval.png'
    # result_sep_plot_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}_sep_eval.png'

    # sep_fig, sep_axs = plt.subplots(dataset.N_TASKS, dataset.N_TASKS, figsize=(40, 40))
    # # tot_fig, tot_axs = plt.subplots(1, dataset.N_TASKS, figsize=(40, 8))
    # fig, axs = plt.subplots(dataset.N_TASKS + 1, dataset.N_TASKS, figsize=(40, 48), sharex='col', sharey='col')
    # define model

    model = get_model(args, device, len(train_loader),
                      dataset.get_transform(args))

    print("model device:", model.device)

    acc_matrix = np.zeros((dataset.N_TASKS, dataset.N_TASKS))
    avg_acc = []
    fwt = []
    bwt = []

    for t in range(dataset.N_TASKS):
        _, _, _ = dataset.get_data_loaders(
            args)
        model_path = os.path.join(
            args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}_{t}.pth")
        print(f'loading from {model_path}')
        # model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
        model.cpu()
        model.net.backbone.load_state_dict({k[9:]:v for k, v in torch.load(model_path)['state_dict'].items() if 'backbone.' in k}, strict=True)
        model.to(device)
        
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=True):
                if args.cl_default:
                    # feature = net(data.cuda(non_blocking=True), return_features=True)
                    feature = model.net.backbone(data.to(device), return_features=True)
                else:
                    # feature = net(data.cuda(non_blocking=True))
                    feature = model.net.backbone(data.to(device))

        if args.train.knn_monitor:
            results = []
            for i in range(len(dataset.test_loaders)):
                # acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
                acc, acc_mask = knn_monitor(model.net.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
                results.append(acc)
                acc_matrix[i][t] = acc
            # results = general_knn_monitor(model.net.backbone, dataset, dataset.memory_loaders, dataset.test_loaders, device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
            # for i in range(len(dataset.test_loaders)):
            #     acc_matrix[i][t] = results[i]
            mean_acc = np.mean(results)
        avg_acc.append(mean_acc)

        # if args.utils.plot_feat:
        #     # tot_axs[t] = plot_feat_tot(
        #     #     model.net.backbone, dataset.memory_loaders, dataset.test_loaders, device, args.cl_default, tot_axs[t])
        #     # for sub_t in range(t + 1):
        #     #     sep_axs[sub_t, t] = plot_feat(model.net.backbone, dataset.memory_loaders,
        #     #                                   dataset.test_loaders, sub_t, device, args.cl_default, sep_axs[sub_t, t])
        #     axs = plot_feat(model.net.backbone, dataset.memory_loaders,
        #                                       dataset.test_loaders, t, device, args.cl_default, axs)
        #     for sub_t in range(t + 1):
        #         sep_axs[sub_t, t] = plot_feat_sep(model.net.backbone, dataset.memory_loaders,
        #                                       dataset.test_loaders, sub_t, device, args.cl_default, sep_axs[sub_t, t])

    if args.utils.posi_trans:
        for t in range(dataset.N_TASKS - 1):
            bwt.append(acc_matrix[t, t] - acc_matrix[t, -1])
        # avg_bwt = np.mean(bwt)
        # bwt.append(avg_bwt)
        # rand_results = []
        # for t in range(1, dataset.N_TASKS):
        #     model = get_model(args, device, len(train_loader), dataset.get_transform(args))
        #     rand_acc, _ = knn_monitor(model.net.backbone, dataset, dataset.memory_loaders[t], dataset.test_loaders[t], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
        #     rand_results.append(rand_acc)
        # for t in range(1, dataset.N_TASKS):
        #     model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}_{t-1}.pth")
        #     print(f'loading from {model_path}')
        #     # model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
        #     model.cpu()
        #     model.net.backbone.load_state_dict({k[9:]:v for k, v in torch.load(model_path)['state_dict'].items() if 'backbone.' in k}, strict=True)
        #     model.to(device)

        #     pre_acc, _ = knn_monitor(model.net.backbone, dataset, dataset.memory_loaders[t], dataset.test_loaders[t], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
        #     fwt.append(pre_acc - rand_results[t-1])

        #   if args.cl_default:
        #     print("evaluation called")
        #     # accs = evaluate(model.net.module.backbone, dataset, device)
        #     accs = evaluate(model.net.backbone, dataset, device)
        #     results.append(accs[0])
        #     results_mask_classes.append(accs[1])
        #     mean_acc = np.mean(accs, axis=1)
        #     if dataset.SETTING == 'class-il':
        #       mean_acc_class_il, mean_acc_task_il = mean_acc
        #       with open(result_recording_path, 'a+') as f:
        #         f.write('\nSCL Accuracy for {} task(s): \t [Class-IL]: {} %'
        #             ' \t [Task-IL]: {} %\n'.format(t + 1, round(
        #           mean_acc_class_il, 2), round(mean_acc_task_il, 2)))
        #     print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

            # torch.save({
            #   'epoch': epoch+1,
            #   'state_dict':model.net.state_dict()
            # }, model_path)
            # print(f"Task Model saved to {model_path}")
            # with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
            #   f.write(f'{model_path}')

            # if hasattr(model, 'end_task'):
            #   model.end_task(dataset)

    # if args.utils.plot_feat:
    #     # tot_fig.tight_layout()
    #     sep_fig.tight_layout()
    #     # tot_fig.savefig(result_plot_tot_path)
    #     sep_fig.savefig(result_sep_plot_path)
    #     fig.tight_layout()
    #     fig.savefig(result_plot_path)

    with open(result_recording_path, 'a+') as f:
        avg_acc_str = np.array2string(np.array(avg_acc), precision=2, separator='\t')
        avg_acc_str = avg_acc_str.replace('[', '').replace(']', '')
        bwt_str = np.array2string(np.array(bwt), precision=2, separator='\t')
        bwt_str = bwt_str.replace('[', '').replace(']', '')
        matrix_str = np.array2string(acc_matrix, precision=2, separator='\t')
        matrix_str = matrix_str.replace('[', '').replace(']', '')
        f.write('\n{}\tavg acc\t{}'.format(args.model.cl_model, avg_acc_str))
        f.write('\n{}\tbwd trans\t{}'.format(args.model.cl_model, bwt_str))
        f.write('\nAccuracy matrix:\n {}\n'.format(matrix_str))

    # with open(result_recording_path, 'a+') as f:
    #     f.write('total time is: %d' % total_time)
    #     f.write("\n")

    # if args.eval is not False and args.cl_default is False:
    #     args.eval_from = model_path


if __name__ == "__main__":
    for i in range(1):
        args = get_args()
        print("device is:", args.device)

        main(device=args.device, args=args)
        # completed_log_dir = args.log_dir.replace(
        #     'in-progress', 'debug' if args.debug else 'completed')
        # os.rename(args.log_dir, completed_log_dir)
        # print(f'Log file has been saved to {completed_log_dir}')

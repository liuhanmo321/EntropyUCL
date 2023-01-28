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
from datasets import get_dataset, get_dataset_off
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
from datasets.seq_tinyimagenet import base_path




def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    # status = model.training
    # model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
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

    # model.train(status)
    return accs, accs_mask_classes

def main(device, args):

    train_loader_off = torch.utils.data.DataLoader(
        dataset=get_dataset_off(dataset=args.dataset.name, data_dir=base_path() + 'CIFAR100',
            transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs), 
            train=True, 
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    test_loader_off = torch.utils.data.DataLoader(
        dataset=get_dataset_off(dataset=args.dataset.name, data_dir=base_path() + 'CIFAR100',
            transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), 
            train=False,
            **args.dataset_kwargs
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )


    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(
        args)

    result_path = os.path.join("./results", args.dataset_kwargs['dataset'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    scl_prefix = '_scl' if args.cl_default else ''
    result_recording_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}{args.utils.comment}_eval.txt'
    result_plot_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}{args.utils.comment}_eval.png'
    result_sep_plot_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}{args.utils.comment}_sep_eval.png'
    # result_recording_path = result_path + '/' + \
    #     args.model.cl_model + scl_prefix + '_eval.txt'
    # result_plot_path = result_path + '/' + args.model.cl_model + scl_prefix + '_eval.png'
    # result_sep_plot_path = result_path + '/' + args.model.cl_model + scl_prefix + '_sep_eval.png'

    sep_fig, sep_axs = plt.subplots(dataset.N_TASKS, dataset.N_TASKS, figsize=(40, 40))
    # tot_fig, tot_axs = plt.subplots(1, dataset.N_TASKS, figsize=(40, 8))
    fig, axs = plt.subplots(dataset.N_TASKS + 1, dataset.N_TASKS, figsize=(40, 48), sharex='col', sharey='col')
    # define model

    model = get_model(args, device, len(train_loader),
                      dataset.get_transform(args))

    print("model device:", model.device)

    acc_matrix = np.zeros((dataset.N_TASKS, dataset.N_TASKS))
    avg_acc = []
    fwt = []
    bwt = []

    if args.method != 'joint':
        # for t in range(dataset.N_TASKS):
            _, _, _ = dataset.get_data_loaders(
                args)
            model_path = os.path.join(
                args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}{args.utils.comment}_{t}.pth")
            print(f'loading from {model_path}')
            # model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
            model.cpu()
            model.net.backbone.load_state_dict({k[9:]:v for k, v in torch.load(model_path)['state_dict'].items() if 'backbone.' in k}, strict=True)
            model.to(device)
            model.eval()

            # if t == dataset.N_TASKS - 1:
            #     global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Linear Training')
            # else:
            #     global_progress = tqdm(range(0, 1), desc=f'Linear Training')
            
            global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Linear Training')

            classifier = nn.Linear(512, 100).to(device)
            optimizer = torch.optim.SGD(classifier.parameters(), args.eval.base_lr,
                                    momentum=args.eval.optimizer.momentum, weight_decay=args.eval.optimizer.weight_decay)
            
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], gamma=0.1)

            for epoch in global_progress:
                running_loss = 0
                classifier.train()
                results, results_mask_classes = [], []
                # for i in range(len(dataset.train_loaders)):
                for idx, (images, labels) in enumerate(train_loader_off):
                    with torch.no_grad():
                        feature = model.net.backbone(images.to(device))
                    preds = classifier(feature)
                    loss = nn.CrossEntropyLoss()(preds, labels.to(device))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    
                    running_loss += loss.item()
                    # data_dict = {"loss", loss}
                
                running_loss = running_loss / len(dataset.train_loaders)

                classifier.eval()
                if epoch % 1 == 0:
                    accs, _ = evaluate(model.net.backbone, dataset, device, classifier) # accs 是每个小数据集的acc
                    mean_acc = np.mean(accs)                            
                    epoch_dict = {"epoch":epoch, "accuracy": mean_acc, 'loss': running_loss}
                    global_progress.set_postfix(epoch_dict)

                if (epoch + 1 == args.eval.num_epochs):
                    accs, _ = evaluate(model.net.backbone, dataset, device, classifier) # accs 是每个小数据集的acc
                    mean_acc = np.mean(accs)                            
                    epoch_dict = {"epoch":epoch, "accuracy": mean_acc, 'loss': running_loss}
                    global_progress.set_postfix(epoch_dict)
                    for i, acc in enumerate(accs):
                        acc_matrix[i][t] = acc

                    avg_acc.append(mean_acc)
                    

            if args.utils.plot_feat:
                axs = plot_feat(model.net.backbone, dataset.memory_loaders,
                                                dataset.test_loaders, t, device, args.cl_default, axs)
                for sub_t in range(t + 1):
                    sep_axs[sub_t, t] = plot_feat_sep(model.net.backbone, dataset.memory_loaders,
                                                dataset.test_loaders, sub_t, device, args.cl_default, sep_axs[sub_t, t])    
    else:
        for t in range(dataset.N_TASKS):
            # train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
            _, _, _ = dataset.get_data_loaders(args)
        
        model_path = os.path.join(
            args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}{args.utils.comment}_{0}.pth")
        print(f'loading from {model_path}')
        # model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
        model.cpu()
        model.net.backbone.load_state_dict({k[9:]:v for k, v in torch.load(model_path)['state_dict'].items() if 'backbone.' in k}, strict=True)
        model.to(device)
        model.eval()

        if args.train.knn_monitor:
            if args.eval.eval_type == 'task':
                results = []
                for i in range(len(dataset.test_loaders)):
                    # acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
                    acc, acc_mask = knn_monitor(model.net.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
                    results.append(acc)
                    acc_matrix[i][t] = acc
            else:                
                results = general_knn_monitor(model.net.backbone, dataset, dataset.memory_loaders, dataset.test_loaders, device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
                for i in range(len(dataset.test_loaders)):
                    acc_matrix[i][t] = results[i]
            mean_acc = np.mean(results)
        avg_acc.append(mean_acc)

        if args.utils.plot_feat:
            # tot_axs[t] = plot_feat_tot(
            #     model.net.backbone, dataset.memory_loaders, dataset.test_loaders, device, args.cl_default, tot_axs[t])
            # for sub_t in range(t + 1):
            #     sep_axs[sub_t, t] = plot_feat(model.net.backbone, dataset.memory_loaders,
            #                                   dataset.test_loaders, sub_t, device, args.cl_default, sep_axs[sub_t, t])
            axs = plot_feat(model.net.backbone, dataset.memory_loaders,
                                            dataset.test_loaders, t, device, args.cl_default, axs)
            for sub_t in range(t + 1):
                sep_axs[sub_t, t] = plot_feat_sep(model.net.backbone, dataset.memory_loaders,
                                            dataset.test_loaders, sub_t, device, args.cl_default, sep_axs[sub_t, t])    

    if args.utils.posi_trans:
        for t in range(dataset.N_TASKS - 1):
            bwt.append(acc_matrix[t, t] - acc_matrix[t, -1])
        
        bwt_list = [0]
        for t in range(1, dataset.N_TASKS):
            temp_bwt = np.array([acc_matrix[t_temp, t_temp] - acc_matrix[t_temp, t] for t_temp in range(t)])
            bwt_list.append(np.mean(temp_bwt))
        bwt_list = np.array(bwt_list)

        new_ds_acc = np.array([acc_matrix[t,t] for t in range(dataset.N_TASKS)])


    if args.utils.plot_feat:
        # tot_fig.tight_layout()
        sep_fig.tight_layout()
        # tot_fig.savefig(result_plot_tot_path)
        sep_fig.savefig(result_sep_plot_path)
        fig.tight_layout()
        fig.savefig(result_plot_path)

    with open(result_recording_path, 'a+') as f:
        avg_acc_str = np.array2string(np.array(avg_acc), precision=2, separator='\t', max_line_width=200)
        avg_acc_str = avg_acc_str.replace('[', '').replace(']', '')
        bwt_str = np.array2string(np.array(bwt), precision=2, separator='\t', max_line_width=200)
        bwt_str = bwt_str.replace('[', '').replace(']', '')
        matrix_str = np.array2string(acc_matrix, precision=2, separator='\t', max_line_width=200)
        matrix_str = matrix_str.replace('[', '').replace(']', '')
        f.write('\n{}\tavg acc\t{}'.format(args.model.cl_model, avg_acc_str))
        f.write('\n{}\tbwd trans\t{}'.format(args.model.cl_model, bwt_str))
        f.write('\nAccuracy matrix:\n {}\n'.format(matrix_str))

        # new_ds_acc_str = np.array2string(np.array(new_ds_acc), precision=2, separator='\t', max_line_width=200)
        # new_ds_acc_str = new_ds_acc_str.replace('[', '').replace(']', '')
        # f.write('\n{}\tnew ds acc\t{}'.format(args.model.cl_model, new_ds_acc_str))

        # bwt_list_str = np.array2string(np.array(bwt_list), precision=2, separator='\t', max_line_width=200)
        # bwt_list_str = bwt_list_str.replace('[', '').replace(']', '')
        # f.write('\n{}\tbwt list\t{}\n\n'.format(args.model.cl_model, bwt_list_str))


if __name__ == "__main__":
    for i in range(1):
        args = get_args()
        print("device is:", args.device)

        main(device=args.device, args=args)
        # completed_log_dir = args.log_dir.replace(
        #     'in-progress', 'debug' if args.debug else 'completed')
        # os.rename(args.log_dir, completed_log_dir)
        # print(f'Log file has been saved to {completed_log_dir}')

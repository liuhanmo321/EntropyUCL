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

from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from augmentations import get_aug


def evaluate(model, dataset: ContinualDataset, device, classifier=None) -> Tuple[list, list]:
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

    if 'cifar100' in args.dataset.name:
        cifar_norm = [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2615]]
        # test_transform = get_aug(train=False, train_classifier=False, mean_std=cifar_norm, **args.aug_kwargs)
        train_dataset = CIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transforms.Compose(
                [transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*cifar_norm)
                ])
        )
        # test_dataset = CIFAR100(base_path() + 'CIFAR100',train=False,
        #                            download=True, transform=transforms.Compose(
        #         [transforms.ToPILImage(),
        #         transforms.RandomCrop(32),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(*cifar_norm)
        #         ])
        # )
        n_class = 100

    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(
        args)

    result_path = os.path.join("./results", args.dataset_kwargs['dataset'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    scl_prefix = '_scl' if args.cl_default else ''
    
    result_recording_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}{args.utils.comment}_linear_eval.txt'

    model = get_model(args, device, len(train_loader), dataset.get_transform(args))
    backbone = model.net.backbone

    backbone.fc = nn.Linear(512, n_class)    
    for name, param in backbone.named_parameters():
        # print(name)
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    backbone.fc.weight.data.normal_(mean=0.0, std=0.01)
    backbone.fc.bias.data.zero_()

    model_path = os.path.join(
                args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}{args.utils.comment}_{dataset.N_TASKS - 1}.pth")
    print(f'loading from {model_path}')
    
    # model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
    backbone.cpu()
    state_dict = torch.load(model_path)['state_dict']
    backbone_dict = {}
    for k, v in state_dict.items():
        if 'backbone.' in k and 'fc.' not in k:
            backbone_dict[k[9:]] = v
    print(backbone_dict.keys())
    backbone.load_state_dict(backbone_dict, strict=False)
    # model.net.backbone.load_state_dict({k[9:]:v for k, v in torch.load(model_path)['state_dict'].items() if 'backbone.' in k}, strict=True)
    backbone.to(device)

    init_lr = args.eval.base_lr * args.eval.batch_size / 256
    criterion = nn.CrossEntropyLoss().cuda(args.device)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, backbone.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.eval.optimizer.momentum,
                                weight_decay=args.eval.optimizer.weight_decay)

    all_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.eval.batch_size, shuffle=True,
        num_workers=args.dataset.num_workers, pin_memory=True)
    
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=args.train.batch_size, shuffle=True,
    #     num_workers=args.dataset.num_workers, pin_memory=True)

    acc_matrix = np.zeros((dataset.N_TASKS, dataset.N_TASKS))
    avg_acc = []
    bwt = []

    for t in range(dataset.N_TASKS):
        _, _, _ = dataset.get_data_loaders(args)

    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Linear Training')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], gamma=0.1)
    
    print('epochs starts')
    for epoch in global_progress:

        backbone.eval()
        running_loss = 0
        results, results_mask_classes = [], []
        # for i in range(len(dataset.train_loaders)):
        for idx, (images, labels) in enumerate(all_train_loader):
            
            images = images.to(args.device)
            labels = labels.to(args.device)

            output = backbone(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # data_dict = {"loss", loss}
        
        # running_loss = running_loss / len(dataset.train_loaders)

        if epoch % 10 == 0:
            accs, _ = evaluate(backbone, dataset, device) # accs 是每个小数据集的acc
            mean_acc = np.mean(accs)                            
            epoch_dict = {"epoch":epoch, "accuracy": mean_acc, 'loss': running_loss}
            global_progress.set_postfix(epoch_dict)

        if (epoch + 1 == args.eval.num_epochs):
            accs, _ = evaluate(backbone, dataset, device) # accs 是每个小数据集的acc
            mean_acc = np.mean(accs)                            
            epoch_dict = {"epoch":epoch, "accuracy": mean_acc, 'loss': running_loss}
            global_progress.set_postfix(epoch_dict)
            for i, acc in enumerate(accs):
                acc_matrix[i][-1] = acc

            avg_acc.append(mean_acc)


    # if args.utils.posi_trans:
    #     for t in range(dataset.N_TASKS - 1):
    #         bwt.append(acc_matrix[t, t] - acc_matrix[t, -1])
        
    #     bwt_list = [0]
    #     for t in range(1, dataset.N_TASKS):
    #         temp_bwt = np.array([acc_matrix[t_temp, t_temp] - acc_matrix[t_temp, t] for t_temp in range(t)])
    #         bwt_list.append(np.mean(temp_bwt))
    #     bwt_list = np.array(bwt_list)

    #     new_ds_acc = np.array([acc_matrix[t,t] for t in range(dataset.N_TASKS)])

    print(f"saving to {result_recording_path}")
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

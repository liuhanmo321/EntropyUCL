import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3, 7"
from arguments import get_args
from importlib_metadata import List
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm

from augmentations import get_aug
from models import get_model
from tools import AverageMeter, general_knn_monitor, knn_monitor, Logger, file_exist_check
from datasets import get_dataset, get_dataset_all
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
import copy
import wandb

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

    dataset_copy = get_dataset(args)
    dataset = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)

    result_path = os.path.join("./results", args.dataset_kwargs['dataset'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    scl_prefix = '_scl' if args.cl_default else ''
    seed = '' if args.seed == None else str(args.seed)
    result_recording_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}{args.utils.comment}.txt'
    result_plot_path = f'{result_path}/{args.model.cl_model}{scl_prefix}_{args.model.name}{args.utils.comment}.png'
 
    fig, axs = plt.subplots(dataset.N_TASKS + 1, dataset.N_TASKS, figsize=(40, 48), sharex='col', sharey='col')
    # define model    
    
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))
    
    # if args.train.parallel:
    #     para_model = nn.DataParallel(model)
    #     # para_model = nn.DataParallel(model, device_ids=[int(args.device[-1]), int(args.device[-1]) + 1])
    #     model = para_model.module

    # print("model device:", model.device)
    del dataset_copy

    logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
      
    accuracy = 0 
    acc_matrix = np.zeros((dataset.N_TASKS, dataset.N_TASKS))
    avg_acc = []
    start_time = time.time()
    if args.model.cl_model != 'joint':
        for t in range(dataset.N_TASKS):
            train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)

            if args.train.more_epochs > 0 and not args.debug:
                global_progress = tqdm(range(0, args.train.stop_at_epoch + args.train.more_epochs), desc=f'Training')
            else:
                global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')

            if ('dis' in args.model.cl_model or 'cassle' in args.model.cl_model):
                if t > 0:
                    if args.train.parallel:
                        model_old = copy.deepcopy(model.net.module.cpu())
                        model_old.encoder = nn.DataParallel((model_old.encoder))
                    else:
                        model_old = copy.deepcopy(model.net.cpu())
                    model_old.to(device)
                model.to(device)

            if args.model.cl_model == 'sep':
                model = get_model(args, device, len(train_loader), dataset.get_transform(args))
            
            # timestamps = []

            for epoch in global_progress:
                model.train()
                results, results_mask_classes = [], []          
                local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
                for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
                    if ('dis' in args.model.cl_model or 'cassle' in args.model.cl_model) and t > 0:
                        data_dict = model.observe(images1, labels, images2, notaug_images, model_old)
                    else:
                        data_dict = model.observe(images1, labels, images2, notaug_images)
                    logger.update_scalers(data_dict)
                
                global_progress.set_postfix(data_dict)

                if args.train.knn_monitor and (epoch % args.train.knn_interval == 0 or (epoch + 1 == args.train.stop_at_epoch)):
                    if args.model.cl_model == 'sep':
                        model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}{args.utils.comment}_{t}.pth")
                        torch.save({
                            'epoch': epoch+1,
                            'state_dict':model.net.state_dict()
                        }, model_path)
                    for i in range(len(dataset.test_loaders)):
                        # acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
                        if args.model.cl_model == 'sep':                      
                            model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}{args.utils.comment}_{i}.pth")
                            model.net.backbone.load_state_dict({k[9:]:v for k, v in torch.load(model_path)['state_dict'].items() if 'backbone.' in k}, strict=True)
                        
                        if args.train.parallel:
                            acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset))) 
                        else:
                            acc, acc_mask = knn_monitor(model.net.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))                             
                        
                        results.append(acc)
                        acc_matrix[i][t] = acc
                    # print("acc calculated at {}".format(epoch))
                    # results = general_knn_monitor(model.net.backbone, dataset, dataset.memory_loaders, dataset.test_loaders, device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
                    # for i in range(len(dataset.test_loaders)):
                    #     acc_matrix[i][t] = results[i]
                    mean_acc = np.mean(results)
                    
                epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
                global_progress.set_postfix(epoch_dict)
                logger.update_scalers(epoch_dict)
            
            avg_acc.append(mean_acc)
        
            if args.utils.plot_feat:
                axs = plot_feat(model.net.backbone, dataset.memory_loaders,
                                        dataset.test_loaders, t, device, args.cl_default, axs)
            # tot_axs[t] = plot_feat_tot(model.net.backbone, dataset.memory_loaders, dataset.test_loaders, device, args.cl_default, tot_axs[t]) 
            # for sub_t in range(t + 1):
            #   sep_axs[sub_t, t] = plot_feat(model.net.backbone, dataset.memory_loaders, dataset.test_loaders, sub_t, device, args.cl_default, sep_axs[sub_t, t])     

        
            # if args.cl_default:
            #     print("evaluation called")
            #     # accs = evaluate(model.net.module.backbone, dataset, device)
            #     accs = evaluate(model.net.backbone, dataset, device)
            #     results.append(accs[0])
            #     results_mask_classes.append(accs[1])
            #     mean_acc = np.mean(accs, axis=1)
            #     if dataset.SETTING == 'class-il':
            #         mean_acc_class_il, mean_acc_task_il = mean_acc
            #         with open(result_recording_path, 'a+') as f:
            #             f.write('\nSCL Accuracy for {} task(s): \t [Class-IL]: {} %'
            #                 ' \t [Task-IL]: {} %\n'.format(t + 1, round(
            #             mean_acc_class_il, 2), round(mean_acc_task_il, 2)))        
            #         print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
    
            model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}{args.utils.comment}_{t}.pth")

            state_dict = model.net.state_dict()
            torch.save({
                'epoch': epoch+1,
                'state_dict': state_dict
            }, model_path)
            print(f"Model of Task {t} saved to {model_path}")
            with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
                f.write(f'{model_path}')       
            
            if hasattr(model, 'end_task'):
                if 'syn' in args.model.cl_model:
                    if t > 0:
                        model.end_task(dataset, model_old)
                elif 'uniform' in args.model.cl_model:
                    if t == 0:
                        model.end_task(dataset)
                    else:
                        model.end_task(dataset, model_old)
                else:
                    model.end_task(dataset)
            
    else:
        for t in range(dataset.N_TASKS):
            # train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
            _, _, _ = dataset.get_data_loaders(args)

        all_train_loader = get_dataset_all(args)    
        global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
        for epoch in global_progress:
            model.train()
            results, results_mask_classes = [], []
            
            local_progress = tqdm(all_train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
            for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
                    data_dict = model.observe(images1, labels, images2, notaug_images)
                    logger.update_scalers(data_dict)
            # for t in range(dataset.N_TASKS):
            #     local_progress=tqdm(dataset.train_loaders[t], desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
            #     for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
            #         data_dict = model.observe(images1, labels, images2, notaug_images)
            #         logger.update_scalers(data_dict)

            global_progress.set_postfix(data_dict)

            if args.train.knn_monitor and (epoch % args.train.knn_interval == 0 or (epoch + 1 == args.train.stop_at_epoch)): 
                for i in range(len(dataset.test_loaders)):
                    # acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
                    acc, acc_mask = knn_monitor(model.net.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset))) 
                    results.append(acc)
                    acc_matrix[i][t] = acc
                # print("acc calculated at {}".format(epoch))
                # results = general_knn_monitor(model.net.backbone, dataset, dataset.memory_loaders, dataset.test_loaders, device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
                # for i in range(len(dataset.test_loaders)):
                #   acc_matrix[i][t] = results[i]
                mean_acc = np.mean(results)
              
            epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
            global_progress.set_postfix(epoch_dict)
            logger.update_scalers(epoch_dict)

            if epoch == 200:
                model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}{'200epoch_'+args.utils.comment}_{0}.pth")
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.net.state_dict()
                }, model_path)
                print(f"Model of Task {0} saved to {model_path}")
                with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
                    f.write(f'{model_path}')
        
        avg_acc.append(mean_acc)
      
        model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}{scl_prefix}_{args.dataset.name}_{args.model.name}{'300epoch_'+args.utils.comment}_{0}.pth")
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.net.state_dict()
        }, model_path)
        print(f"Model of Task {0} saved to {model_path}")
        with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
            f.write(f'{model_path}')

        if hasattr(model, 'end_task'):
            model.end_task(dataset) 
    
    # tot_fig.tight_layout()
    # sep_fig.tight_layout()
    # tot_fig.savefig(result_plot_tot_path)
    # sep_fig.savefig(result_plot_path)
    if args.utils.plot_feat:
        fig.tight_layout()
        fig.savefig(result_plot_path)
    
    with open(result_recording_path, 'a+') as f:
        avg_acc_str = np.array2string(np.array(avg_acc), precision=2, separator='\t', max_line_width=200)
        avg_acc_str = avg_acc_str.replace('[', '').replace(']', '')
        matrix_str = np.array2string(acc_matrix, precision=2, separator='\t', max_line_width=200)
        matrix_str = matrix_str.replace('[', '').replace(']', '')
        f.write('\n{}\t{}'.format(args.model.cl_model, avg_acc_str))
        f.write('\nAccuracy matrix:\n {}\n'.format(matrix_str))
    # np.savetxt(result_recording_path, acc_matrix, delimiter="\t")
    
    end_time = time.time()
    total_time = end_time - start_time    
    with open(result_recording_path, 'a+') as f:
        f.write('total time is: %d' % total_time)
        f.write("\n")
    
    wandb.finish()
    # if args.eval is not False and args.cl_default is False:
    #     args.eval_from = model_path

if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(visible_devices)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3" 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5" 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7" 
    for i in range(1):    
        args = get_args()    
        print("device is:", args.device)

        # args.device = 1

        main(device=args.device, args=args)
        completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
        os.rename(args.log_dir, completed_log_dir)
        print(f'Log file has been saved to {completed_log_dir}')



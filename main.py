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
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from datetime import datetime
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from typing import Tuple
import time

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
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)

    result_path = os.path.join("./results", args.dataset_kwargs['dataset'])
    if not os.path.exists(result_path):
      os.makedirs(result_path)

    # define model
    
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))

    print("model device:", model.device)

    logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0 

    start_time = time.time()
    for t in range(dataset.N_TASKS):
      train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
      global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
      for epoch in global_progress:
        model.train()
        results, results_mask_classes = [], []
        
        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
        for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
            data_dict = model.observe(images1, labels, images2, notaug_images)
            logger.update_scalers(data_dict)

        global_progress.set_postfix(data_dict)

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
            for i in range(len(dataset.test_loaders)):
              acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset))) 
              results.append(acc)
            mean_acc = np.mean(results)
          
        epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
      
      with open(result_path + '/' + args.model.cl_model + '.txt', 'a+') as f:
        f.write('\nAccuracy for {} task(s): \t [Class-IL]: {} %\n'.format(t + 1, round(
          mean_acc, 2)))

      if args.model.cl_model == 'ours':
        model.cluster()
     
      if args.cl_default:
        print("evaluation called")
        accs = evaluate(model.net.module.backbone, dataset, device)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        mean_acc = np.mean(accs, axis=1)
        if dataset.SETTING == 'class-il':
          mean_acc_class_il, mean_acc_task_il = mean_acc
          with open(result_path + '/' + args.model.cl_model + '.txt', 'a+') as f:
            f.write('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
                ' \t [Task-IL]: {} %\n'.format(t + 1, round(
              mean_acc_class_il, 2), round(mean_acc_task_il, 2)))        
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
 
      model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}.pth")
      torch.save({
        'epoch': epoch+1,
        'state_dict':model.net.state_dict()
      }, model_path)
      print(f"Task Model saved to {model_path}")
      with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')
      
      if hasattr(model, 'end_task'):
        model.end_task(dataset)
    
    end_time = time.time()
    total_time = end_time - start_time    
    with open(result_path + '/' + args.model.cl_model + '.txt', 'a+') as f:
      f.write('total time is: %d' % total_time)
      f.write("\n")

    if args.eval is not False and args.cl_default is False:
        args.eval_from = model_path

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(visible_devices)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5" 
    # os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7" 
    args = get_args()
   
    print("device is:", args.device)
    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')



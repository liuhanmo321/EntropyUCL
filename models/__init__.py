import os
import importlib
from .simsiam import SimSiam
from .barlowtwins import BarlowTwins
from torchvision.models import resnet50, resnet18
import torch
from .backbones import resnet18

def get_backbone(backbone, dataset, castrate=True, cm_model='simsiam'):
    backbone = eval(f"{backbone}()")
    if dataset == 'seq-cifar100':
        backbone.n_classes = 100
    elif dataset == 'seq-cifar10':
        backbone.n_classes = 10
    
    if 'cifar10' in dataset:
        backbone.n_classes = 10    
    if 'cifar100' in dataset:
        backbone.n_classes = 100
    if 'mnist' in dataset:
        backbone.n_classes = 10

    # When unsupervised, the fc layer is turned to Idnetity matrix.
    # mult_const = 2 if cm_model == 'barlowtwins' else 1
    # print(f"the rep dimension mult const is {mult_const}")
    backbone.output_dim = backbone.fc.in_features
    if not castrate:
        backbone.fc = torch.nn.Identity()

    return backbone


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('models')
            if not model.find('__') > -1 and 'py' in model]

def get_model(args, device, len_train_loader, transform):
    loss = torch.nn.CrossEntropyLoss()
    if args.model.name == 'simsiam':
        backbone =  SimSiam(get_backbone(args.model.backbone, args.dataset.name, args.cl_default)).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)
    elif args.model.name == 'barlowtwins':
        backbone = BarlowTwins(get_backbone(args.model.backbone, args.dataset.name, args.cl_default, args.model.name), device).to(device)
        if args.model.proj_layers is not None:
            backbone.projector.set_layers(args.model.proj_layers)

    names = {}
    # print(get_all_models())
    for model in get_all_models():
        mod = importlib.import_module('models.' + model)
        # class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
        # print({x.lower():x for x in mod.__dir__()})
        # if model == 'ours_sep_replay':
        #     print({x.lower():x for x in mod.__dir__()})
        class_name = {x.lower():x for x in mod.__dir__()}[model]
        names[model] = getattr(mod, class_name)
    
    return names[args.model.cl_model](backbone, loss, args, len_train_loader, transform)


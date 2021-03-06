'''
import and prepare the dataset

reference:
https://github.com/ae-foster/pytorch-simclr
'''
import json

import torchvision
import torchvision.transforms as transforms

from augmentation import ColourDistortion
from dataset import *
from models import *


def get_datasets(dataset, augment_clf_train=False, add_indices_to_data=False, num_positive=None):

    CACHED_MEAN_STD = {
        'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        'imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    }

    PATHS = {
        'cifar10': '/data/cifar10/',
        'imagenet': '/data/imagenet/2012/'
    }
    try:
        with open('dataset-paths.json', 'r') as f:
            local_paths = json.load(f)
            PATHS.update(local_paths)
    except FileNotFoundError:
        pass
    root = PATHS[dataset]

    # Data
    if dataset == 'imagenet':
        img_size = 224
    else:
        img_size = 32

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        ColourDistortion(s=0.5),
        transforms.ToTensor(),
        transforms.Normalize(*CACHED_MEAN_STD[dataset]),
    ])

    if dataset == 'imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])

    if augment_clf_train:
        transform_clftrain = transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*CACHED_MEAN_STD[dataset]),
        ])
    else:
        transform_clftrain = transform_test

    if dataset == 'cifar10':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.CIFAR10)
        else:
            dset = torchvision.datasets.CIFAR10
        if num_positive is None:
            trainset = CIFAR10Biaugment(root=root, train=True, download=True, transform=transform_train)
        else:
            trainset = CIFAR10Multiaugment(root=root, train=True, download=True, transform=transform_train,
                                           n_augmentations=num_positive)
        testset = dset(root=root, train=False, download=True, transform=transform_test)
        clftrainset = dset(root=root, train=True, download=True, transform=transform_clftrain)
        num_classes = 10
        stem = StemCIFAR
    elif dataset == 'imagenet':
        if add_indices_to_data:
            dset = add_indices(torchvision.datasets.ImageNet)
        else:
            dset = torchvision.datasets.ImageNet
        if num_positive is None:
            trainset = ImageNetBiaugment(root=root, split='train', transform=transform_train)
        else:
            raise NotImplementedError
        testset = dset(root=root, split='val', transform=transform_test)
        clftrainset = dset(root=root, split='train', transform=transform_clftrain)
        num_classes = len(testset.classes)
        stem = StemImageNet
    else:
        raise ValueError("Bad dataset value: {}".format(dataset))

    return trainset, testset, clftrainset, num_classes, stem

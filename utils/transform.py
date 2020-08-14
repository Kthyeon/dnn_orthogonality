import torch
import torch.nn
from torch import cuda
from torchvision import datasets, transforms

import random
import os
import ast
from thop import profile
from torch.utils.data import DataLoader, sampler, Subset


def train_valid(_train_set, ratio):
    # ratio indicates the size of validation set. 0.0 indicates using full train dataset.
    # 0.3 indicates using 0.3 portion of the entire training set.
    assert 0. <= ratio <1.0, "Valid ratio should be in the range from 0.0 to 1.0." 
    
    dic = {}
    for i in range(len(_train_set.classes)):
        dic[str(i)] = []

    for id, data in enumerate(_train_set):
        dic[str(data[1])].append(id)

    train_list = []
    valid_list = []
    
    for key in dic.keys():
        tmp = random.sample(dic[key], int(round(len(_train_set) / len(_train_set.classes) * ratio, 2)))
        train_list += (list(set(dic[key]) - set(tmp)))
        valid_list += tmp

    return train_list, valid_list

def mean_list(data='cifar10'):
    try:
        if data == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            img_size = 32
        elif data == 'cifar100':
            mean = [0.5071, 0.4865, 0.4409]
            std = [0.2673, 0.2564, 0.2762]
            img_size = 32
        elif data == 'tiny-image':
            mean = [0.4802, 0.4481, 0.3975]
            std = [0.2770, 0.2691, 0.2821]
            img_size = 64
        elif data == 'image':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            img_size = 224
        else:
            raise Exception('There is no dataset, here. Only cifar, tiny-imagenet, imagenet')
    except Exception as e:
        print(e)

    return mean, std, img_size


def transform_setter(dataset='cifar10'):
    mean, std, img_size = mean_list(data = dataset)
    # train, test transform function
    if dataset != 'image':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean =mean, std = std)
            ])
        test_transforms = transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize(mean = mean, std = std)
            ])
    else:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
            ])
        test_transforms = transforms.Compose([
                                transforms.Resize(size=256),
                                transforms.CenterCrop(size=img_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = mean, std = std)
                            ])
    
    return train_transforms, test_transforms

def data_setter(args, root = '/home/taehyeon/'):

    train_transforms, test_transforms = transform_setter(dataset = args.dataset)
    if args.dataset == 'cifar10':
        _trainset = datasets.CIFAR10(root + args.dataset + '/', train = True, transform = train_transforms, download = True)
        testset = datasets.CIFAR10(root + args.dataset + '/', train = False, transform = test_transforms, download = False)
    elif args.dataset == 'cifar100':
        _trainset = datasets.CIFAR100(root + args.dataset + '/', train = True, transform = train_transforms, download = True)
        testset = datasets.CIFAR100(root + args.dataset + '/', train = False, transform = test_transforms, download = False)
    elif args.dataset == 'tiny-image':
        _trainset = datasets.ImageFolder(root + args.dataset + '/train', transform=train_transforms)
        testset = datasets.ImageFolder(root + args.dataset + '/val', transform=test_transforms)
    elif args.dataset == 'image':
        _trainset = datasets.ImageFolder(root + args.dataset + '/Data/train', transform=train_transforms)
        testset = dataset.ImageFolder(root + args.dataset + '/Data/valid', transform=test_transforms)

    train_list, valid_list = train_valid(_trainset, args.valid_size)
    trainset = Subset(_trainset, train_list)
    validset = Subset(_trainset, valid_list)
    
    # Generate the dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle=True, pin_memory = args.pin_memory, num_workers = args.num_workers)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size = args.batch_size, pin_memory = args.pin_memory, num_workers = args.num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size, pin_memory = args.pin_memory, num_workers = args.num_workers)
    
    dataloader = {
            'train': train_loader,
            'valid': valid_loader,
            'test': test_loader
            }

    dataset_size = {'train': len(trainset), 'valid': len(validset), 'test': len(testset)}

    return dataloader, dataset_size

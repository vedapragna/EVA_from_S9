# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:07:34 2020

Updated on Wed Apr 8

@author: Vedapragna
"""
import torch
import torchvision

import torch.nn as nn
import torch.optim as optim

from albumentations.pytorch import ToTensor
from albumentations import Normalize, HorizontalFlip, Cutout, Compose, PadIfNeeded, RandomCrop
from albumentations.imgaug.transforms import IAAFliplr
import numpy as np

#Normalise_param, pad_param, RandomCrop_param, cutout_param, Fliplr_prob

class album_transform_v1:
    def __init__(self,Normalise_param = [], pad_param = [], RandomCrop_param = [], cutout_param = [], Fliplr_prob = 0):

        tranform_lst = []

        if Normalise_param:
            mn = Normalise_param[0]
            sd = Normalise_param[1]
            tranform_lst.append(Normalize((mn, mn, mn), (sd, sd, sd)))

        if pad_param:
            height = pad_param[0]
            width = pad_param[1]
            shud_apply = pad_param[2]
            pad_prob = pad_param[3]
            tranform_lst.append(PadIfNeeded(min_height= height,min_width=width,always_apply=shud_apply, p=pad_prob))

        if RandomCrop_param:
            crop_ht = RandomCrop_param[0]
            crop_wdt = RandomCrop_param[1]
            shud_apply = RandomCrop_param[2]
            crop_prob = RandomCrop_param[3]
            tranform_lst.append(RandomCrop(crop_ht, crop_wdt, always_apply= shud_apply, p = crop_prob))

        if cutout_param:
            holes = cutout_param[0]
            hole_ht = cutout_param[1]
            hole_wdt = cutout_param[2]
            cutout_prob = cutout_param[3]
            tranform_lst.append(Cutout(num_holes=holes, max_h_size=hole_ht, max_w_size = hole_wdt, p = cutout_prob))

        if Fliplr_prob > 0:
            tranform_lst.append(IAAFliplr(p = Fliplr_prob))

        tranform_lst.append(ToTensor()) 

        self.transform = Compose(tranform_lst)

    
    def __call__(self,img):
        img = np.array(img)
        img = self.transform(image = img)['image']
        return img

def Get_Cifar10(train_aug_dict, test_aug_dict):
    
    traintransform = album_transform_v1(Normalise_param = train_aug_dict['Normalise'], 
                                     pad_param = train_aug_dict['pad'],
                                     RandomCrop_param = train_aug_dict['Randomcrop'],
                                     cutout_param = train_aug_dict['cutout'],
                                     Fliplr_prob = train_aug_dict['Fliplr'])
    
    testtransform = album_transform_v1(Normalise_param = train_aug_dict['Normalise'])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=traintransform)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=testtransform)
    
    return trainset,testset

class album_transform:
    def __init__(self,flag):
        self.traintransform = Compose([HorizontalFlip(),
                                       Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       Cutout(num_holes=1, max_h_size=4, max_w_size=4, p =0.3),
                                       ToTensor()])
        self.testtransform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ToTensor()])
        self.flag = flag
    
    def __call__(self,img):
        img = np.array(img)
        if self.flag == "train":
            img = self.traintransform(image = img)['image']
        else:
            img = self.testtransform(image = img)['image']
        return img

def Get_Cifar():
    
    traintransform = album_transform("train")
    testtransform = album_transform("test")
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=traintransform)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=testtransform)
    
    return trainset,testset


def DataLoader(train_set,test_set,cuda_batch_size,SEED):
    
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=cuda_batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
    
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

    return train_loader, test_loader

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device



def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
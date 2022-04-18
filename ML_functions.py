#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:23:07 2022

@author: osboxes
"""
import os
import numpy as np
import torch as th




def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed: posibly due to they were already created." % path)
    else:
        print ("Successfully created the directory %s" % path)

def square_image_transformation(imgs):
    # Zero padding for images
    sensors_shape = imgs.shape[2]*imgs.shape[3]

    sqrt = np.sqrt(sensors_shape)
    if not(sqrt.is_integer()):
        size = int(sqrt) + 1
    else:
        size = sqrt

    # Flatten sensors
    imgs = np.reshape(imgs, (imgs.shape[0], sensors_shape))
    aux_current_images = np.ones((imgs.shape[0], size * size))
    aux_current_images[:,:sensors_shape] = imgs[:,:sensors_shape]

    # Squaring the sensors
    X = np.reshape(aux_current_images, (imgs.shape[0], 1, size, size))

    return X

def ML_model_read_func (nnet, nnet_file, dataset, epoch, seed, noise, nnet_dir = 'neural_network_stored/'):
    device = th.device("cpu")

    if(nnet == 'resnet'):
        model = nnet_file.ResNet18(img_channels=1, num_classes = 5).to(device)
        # model = ResNet_C_Github.ResNet50(num_classes = 5, channels= 1).to(device) 
    else:
        layers = 10
        growth = 24
        bottleneck = True
        reduce = 1.0
        model = nnet_file.DenseNet3(layers, num_classes = 5, growth_rate = growth, reduction=reduce,
                                 bottleneck=bottleneck, dropRate=0.05).to(device)

    

    print("Loading model...")
    model.load_state_dict(th.load("{}_C_{}_{}_s{}_model_epoch{}.pth".format(nnet_dir + nnet,dataset,noise, seed,epoch), map_location=th.device('cpu')))
    return model
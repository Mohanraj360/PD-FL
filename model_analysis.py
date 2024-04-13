# %%
# Lab 10 MNIST and softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as dsets
import random

import numpy as np
import os
import datetime
import time
import multiprocessing
import copy
import re
import argparse
import yaml

from itertools import product
from unicodedata import name
from sklearn import datasets
import matplotlib.pyplot as plt

from ripser import Rips
# from persim import PersistenceImager

import seaborn
import pandas as pd

from memory_profiler import profile
import psutil
import traceback

from model_analysis_nets import *
from model_analysis_utils import *
from model_analysis_args import arg_set


torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

VR = []


def para_calc(i, ns, nb, tensors):
    vr = np.linspace(
        0, 0, tensors[0].shape[0] ** 2).reshape(tensors[0].shape[0], tensors[0].shape[0])
    if i % 128 == 0:
        print(i)
    for j in range(i+1, len(ns)):
        for nps in tensors:
            vr[i, j] += (nps[i] - nb[i]) * (nps[j] - nb[j])
        # vp[i,j] /= (ns[i] * ns[j])
        vr[i, j] = vr[i, j] / ((ns[i] * ns[j]) + int(not (ns[i] * ns[j])))
    return vr


def para_get(value):
    # print(value)
    global VR
    VR.append(value)
    return


# %%
dgms = []
H0_dgm = []
H1_dgm = []
# %%


def gridsize(h_gdm):
    s = [999, 0, 999, 0]
    for h in h_gdm:
        s[0] = s[0] if h[0] > s[0] else h[0]
        s[1] = s[1] if h[0] < s[1] else h[0]
        s[2] = s[2] if h[1] > s[2] else h[1]
        s[3] = s[3] if h[1] < s[3] else h[1]
    return s


def makegrid(h_gdm, s):
    height = (s[1] - s[0]) / 126
    weight = (s[3] - s[2]) / 126
    grid = np.zeros((128, 128))
    for h in h_gdm:
        x = y = 0
        while x < 126:
            if (h[0] < s[0] + x * height):
                break
            x += 1
        while y < 126:
            if (h[1] < s[2] + y * weight):
                break
            y += 1
        grid[x][y] += 1
    return grid


def get_rips(value):
    global dgms
    global H0_dgm
    global H1_dgm
    dgms.append(value)
    H0_dgm.append((value[0][0], value[1]))
    H1_dgm.append((value[0][1], value[1]))
    # print("added")
    # print("len(H1_dgm) = {}".format(len(H1_dgm)))


def parallel(d, idx):
    global dgms
    global H0_dgm
    global H1_dgm
    if idx % 10 == 0:
        print(idx)
    rips = Rips(maxdim=1, thresh=100, verbose=False)
    print("calculating", end="")
    for i in range(999):
        if psutil.virtual_memory().percent < 70:
            break
        if i % 10 == 0:
            print("'sleeping {}'".format(i), end="")
        time.sleep(1)
    a = rips.fit_transform(d[0], distance_matrix=True)
    print("'calculated with a.len={}'\n".format(len(a[1])), end="")
    value = (a, d[1])
    return value


def errorcallback(args):
    try:
        print("ERROR0:", args)
    except:
        print("ERROR1:", traceback.format_exc(3))



if __name__ == '__main__':
    # %%
    # global VR
    
    args = arg_set()


    if args.sleep > 0:
        print(f"sleeping:{args.sleep}")
        time.sleep(args.sleep)

    if args.check_usage:
        while(psutil.cpu_percent() > 30):
            print(f"cpu: {psutil.cpu_percent()}sleep 60 sec")
            time.sleep(60)

    t_r = args.thread_rate
    md = args.model
    # round_range = [i for i in range(int(re.findall(
    #     r'\d+', args.round_range)[0]), int(re.findall(r'\d+', args.round_range)[1]))]
    # number_range = [i for i in range(int(re.findall(
    #     r'\d+', args.number_range)[0]), int(re.findall(r'\d+', args.number_range)[1]))]
    round_range = range(args.round_range[0], args.round_range[1])
    number_range = range(args.number_range[0], args.number_range[1])
    # print(round_range)

    pth_d = args.pth_d
    pth_m = args.pth_m
    pth_p = args.pth_p
    pth_r = args.pth_r
    pth_s = args.pth_s
    t = args.t

    print(args.__dict__)
    print(datetime.datetime.now().strftime("%m-%d--%H-%M-%S"))
    
    with open(f'logs/settings-{pth_p}.txt', 'w') as f:
        for eachArg, value in args.__dict__.items():
            f.writelines(" --"+eachArg + ' ' + str(value) + '\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for reproducibility
    random.seed(111)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    learning_rate = 0.001
    training_epochs = 5
    batch_size = 128

    # %%
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform1 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
    trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
    trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
    trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                std=[0.267, 0.256, 0.276])])
    trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                std=[0.267, 0.256, 0.276])])

    GTSRB_data_transforms = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # %%
    if md == "vgg":
        testset = torchvision.datasets.CIFAR10(root='CIFAR10_data/', train=False,
                                               download=True, transform=trans_cifar10_train)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  drop_last=True, shuffle=True, num_workers=2)

        model = VGG16().to(device)
    elif md == "lenet":
        mnist_test = dsets.MNIST(root='MNIST_data/',
                                 train=False,
                                 transform=trans_mnist,
                                 download=True)
        test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True, num_workers=2)
        model = LeNet().to(device)
    elif md == "mnistnet":
        mnist_test = dsets.MNIST(root='MNIST_data/',
                                 train=False,
                                 transform=trans_mnist,
                                 download=True)
        test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True, num_workers=2)
        model = MnistNet().to(device)
    elif md == "resnet":
        testset = torchvision.datasets.CIFAR10(root='CIFAR10_data/', train=False,
                                               download=True, transform=trans_cifar10_train)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  drop_last=True, shuffle=True, num_workers=2)

        model = resnet20().to(device)
    else:
        print("WRONG MODEL")
        assert 0 == 1
    print(model)
    
    if not args.skip_metric:

        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # %%
        # if args.pth:
        #     dir_pth = os.path.join(args.pth, args.pth_p)
        if "LG-FedAvg" in args.pth:
            dir_pth = os.path.join(args.pth, args.pth_r, args.pth_d, args.pth_m, args.pth_s, args.pth_p)
            modelset = []
            modelnames = []
            dir = os.listdir(os.path.join(dir_pth, "local_attack_save"))
            for mt in dir:
                if 'iter' in mt:
                    modelnames.append(mt)
            # print(modelnames)
            for mn in modelnames:
                if int(re.findall(r'\d+', mn)[1]) in number_range and int(re.findall(r'\d+', mn)[0]) in round_range:
                    modelset.append([os.path.join(dir_pth, "local_attack_save"), mn])

            modelnames = []
            dir = os.listdir(os.path.join(dir_pth, "local_normal_save"))
            for mt in dir:
                if 'iter' in mt:
                    modelnames.append(mt)
            # print(modelnames)
            for mn in modelnames:
                if int(re.findall(r'\d+', mn)[1]) in number_range and int(re.findall(r'\d+', mn)[0]) in round_range:
                    modelset.append([os.path.join(dir_pth, "local_normal_save"), mn])

        else:
            # dir = os.listdir(os.path.join(dir_pth, "local_attack_save"))
            dir = os.listdir(dir_pth)
            # dir_normal = os.listdir(os.path.join(dir_pth, "local_normal_save"))
            modelset = []
            modelnames = []
            for mt in dir:
                if 'local' in mt:
                    modelnames.append(mt)
            # print(modelnames)
            for mn in modelnames:
                if int(re.findall(r'\d+', mn)[1]) in number_range and int(re.findall(r'\d+', mn)[0]) in round_range:
                    modelset.append([dir_pth, mn])

        # modelnames = []
        # for mt in dir_normal:
        #     if(mt[0:len("iter_")] == "iter_"):
        #         modelnames.append(mt)
        # for mn in modelnames:
        #     if int(re.findall(r'\d+', mn)[0]) in round_range and int(re.findall(r'\d+', mn)[1][:-1]) in number_range:
        #         modelset.append(["/local_normal_save/", mn])
        # print(dir)
        now = datetime.datetime.now()
        _time = now.strftime("%m-%d--%H-%M-%S")
        if md == "vgg":
            if not os.path.exists(os.path.join(args.save_pth, args.pth_p)):
                os.makedirs(os.path.join(args.save_pth, args.pth_p), exist_ok=True)
            savepath = os.path.join(args.save_pth, args.pth_p, "vr_metric-single-cifar_vgg=")
                
        elif md == "lenet":
            if not os.path.exists(f'./metric_fed_eval_{pth_p}'):
                os.makedirs(f'./metric_fed_eval_{pth_p}', exist_ok=True)
            savepath = f'./metric_fed_eval_{pth_p}' + \
                "/vr_metric-single-mnist_lenet="
        elif md == "mnistnet":
            if not os.path.exists(os.path.join(args.save_pth, args.pth_p)):
                os.makedirs(os.path.join(args.save_pth, args.pth_p), exist_ok=True)
            savepath = os.path.join(args.save_pth, args.pth_p, "vr_metric-single-mnist_mnistnet=")
        elif md == "resnet":
            if not os.path.exists(os.path.join(args.save_pth, args.pth_p)):
                os.makedirs(os.path.join(args.save_pth, args.pth_p), exist_ok=True)
            savepath = os.path.join(args.save_pth, args.pth_p, "vr_metric-single-cifar_resnet=")
        else:
            print("WRONG MODEL")
            assert 0 == 1
        # %%
        print(len(modelset))
        start = time.time()
        print(f"use trhead:{int(multiprocessing.cpu_count()*t_r)-1}, useage:{psutil.cpu_percent()}")

        # %%
        for idx, (modelpth, modelname) in enumerate(modelset):
            elapsed = time.time() - start
            print(
                f"progress:{idx/len(modelset)*100}%, eta:{elapsed *(len(modelset)/(idx or 1)-1)} sec")
            print([modelpth, modelname])
            parm = {}
            if args.state_dict:
                model.load_state_dict(torch.load(os.path.join(modelpth,modelname))['state_dict'])
            else:
                model.load_state_dict(torch.load(os.path.join(modelpth,modelname)))
            print(f"t={t}")
#             t = 50
            if md == "vgg":
                tensor_group = evaulation_vgg(model, test_loader, t, batch_size)
            elif md == "lenet":
                tensor_group = evaulation_lenet(model, test_loader, t, batch_size)
            elif md == "mnistnet":
                tensor_group = evaulation_mnist(model, test_loader, t, batch_size)
            elif md == "resnet":
                tensor_group = evaulation_resnet(model, test_loader, t, batch_size)
            else:
                print("WRONG MODEL")
                assert 0 == 1
            l = 0
            for tensors in tensor_group:
                nb = np.linspace(0, 0, tensors[0].shape[0])
                print(nb.shape)
                for i in tensors:
                    nb += i
                nb /= t

                ns = np.linspace(0, 0, tensors[0].shape[0])
                vr = np.linspace(
                    0, 0, tensors[0].shape[0] ** 2).reshape(tensors[0].shape[0], tensors[0].shape[0])

                for i in range(0, len(ns)):
                    for nps in tensors:
                        ns[i] += nps[i] ** 2
                    ns[i] = ns[i] ** 0.5
                print(f"use thread:{int(multiprocessing.cpu_count()*t_r)-1}, useage:{psutil.cpu_percent()}")
                with multiprocessing.Pool(int(multiprocessing.cpu_count()*t_r)-1) as pool:
                    for i in range(0, len(ns)):
                        pool.apply_async(
                            para_calc, (i, ns, nb, tensors), callback=para_get)
                    pool.close()
                    pool.join()

                for v in VR:
                    vr += v

                VR = []
                dt = datetime.datetime.now()
                np.savetxt(savepath+modelname+'_'+str(l)+".dat", vr+vr.T, fmt="%1.5f")
                print(savepath+modelname+'_'+str(l)+".dat")
                l += 1
#%%
    if md == "vgg":
        # dir = os.listdir(f"./metric_fed_vgg_eval_{pth_p}")
        # metric_path = f"./metric_fed_vgg_eval_{pth_p}/"

        # if not os.path.exists(f'./grids_fed_vgg_eval_{pth_p}/'):
        #     os.makedirs(f'./grids_fed_vgg_eval_{pth_p}/', exist_ok=True)
        # savepth = f"./grids_fed_vgg_eval_{pth_p}/"
        dir = os.listdir(os.path.join(args.save_pth, args.pth_p))
        metric_path = os.path.join(args.save_pth, args.pth_p)
        if not os.path.exists(os.path.join(args.grid_save_pth,args.pth_p)):
            os.makedirs(os.path.join(args.grid_save_pth,args.pth_p), exist_ok=True)
        savepth = os.path.join(args.grid_save_pth,args.pth_p)
    elif md == "lenet":
        dir = os.listdir(f"./metric_fed_eval_{pth_p}")
        metric_path = f"./metric_fed_eval_{pth_p}/"

        if not os.path.exists(f'./grids_fed_eval_{pth_p}/'):
            os.makedirs(f'./grids_fed_eval_{pth_p}/', exist_ok=True)
        savepth = f"./grids_fed_eval_{pth_p}/"
    elif md == "mnistnet":
        dir = os.listdir(os.path.join(args.save_pth, args.pth_p))
        metric_path = os.path.join(args.save_pth, args.pth_p)
        if not os.path.exists(os.path.join(args.grid_save_pth,args.pth_p)):
            os.makedirs(os.path.join(args.grid_save_pth,args.pth_p), exist_ok=True)
        savepth = os.path.join(args.grid_save_pth,args.pth_p)
    elif md == "resnet":
        dir = os.listdir(os.path.join(args.save_pth, args.pth_p))
        metric_path = os.path.join(args.save_pth, args.pth_p)
        if not os.path.exists(os.path.join(args.grid_save_pth,args.pth_p)):
            os.makedirs(os.path.join(args.grid_save_pth,args.pth_p), exist_ok=True)
        savepth = os.path.join(args.grid_save_pth,args.pth_p)

    else:
        print("WRONG MODEL")
        assert 0 == 1

    data = []

#%%
    for metric in dir:
        if "iter" in metric and int(re.findall(r'\d+', metric)[1]) in number_range:
            data.append([np.loadtxt(os.path.join(metric_path,metric)), metric])
    print(len(data))

    print("pre-processing")
    for i in range(len(data)):
        data[i][0] = abs(data[i][0]) * 100
        test = [j for j in data[i][0][0:-1] if j.any() != 0]
        test = np.array(test)
        test = [j for j in test.T[:, 0:-1] if j.any() != 0]
        test = np.array(test)
        if test.shape[0] > test.shape[1]:
            test = test[0:test.shape[1], :]
        elif test.shape[0] < test.shape[1]:
            test = test[:, 0:test.shape[0]]
        data[i][0] = np.array(test)

#%%
    print("multiprocessing")
    start = time.time()
    multiprocessing.freeze_support()
    for top in range(0, len(data)-1, 80):
        elapsed = time.time() - start
        print(
            f"progress:{top/len(data)*100}%, eta:{elapsed *(len(data)/(top or 1)-1)} sec")
        print(top)
        print(f"use trhead:{int(multiprocessing.cpu_count()*t_r)-1}, useage:{psutil.cpu_percent()}")
        with multiprocessing.Pool(int(multiprocessing.cpu_count()*t_r)-1) as pool:
            for idx, d in enumerate(data[top:min(top+80, len(data))]):
                pool.apply_async(
                    parallel, (d, idx,), callback=get_rips, error_callback=errorcallback)

            pool.close()
            pool.join()

        print("len(H1_dgm) = {}".format(len(H1_dgm)))

        for h1 in H1_dgm:
            np.savetxt(os.path.join(savepth,"h_"+h1[1]), makegrid(h1[0], gridsize(h1[0])))
            print(os.path.join(savepth,"h_"+h1[1]))

        dgms = []
        H0_dgm = []
        H1_dgm = []

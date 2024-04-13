# %%
# Lab 10 MNIST and softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import random
import datetime
import os
import sys

import numpy as np
import copy

import multiprocessing

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


if __name__ == '__main__':
    # %%
    # global VR
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for reproducibility
    random.seed(111)
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)

    # %%
    # parameters
    learning_rate = 0.001
    training_epochs = 5
    batch_size = 50
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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # %%
    testset = torchvision.datasets.CIFAR10(root='CIFAR10_data/', train=False,
                                        download=True, transform=transform1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # %%
    class MyNet(nn.Module):

        def __init__(self):
            super(MyNet,self).__init__()
            self.conv1 = nn.Conv2d(3,64,3,padding=1)
            self.conv2 = nn.Conv2d(64,64,3,padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()

            self.conv3 = nn.Conv2d(64,128,3,padding=1)
            self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
            self.pool2 = nn.MaxPool2d(2, 2, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.relu2 = nn.ReLU()

            self.conv5 = nn.Conv2d(128,128, 3,padding=1)
            self.conv6 = nn.Conv2d(128, 128, 3,padding=1)
            self.conv7 = nn.Conv2d(128, 128, 1,padding=1)
            self.pool3 = nn.MaxPool2d(2, 2, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu3 = nn.ReLU()

            self.conv8 = nn.Conv2d(128, 256, 3,padding=1)
            self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
            self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
            self.pool4 = nn.MaxPool2d(2, 2, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.relu4 = nn.ReLU()

            self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
            self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
            self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
            self.pool5 = nn.MaxPool2d(2, 2, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.relu5 = nn.ReLU()

            self.fc14 = nn.Linear(512*4*4,1024)
            self.drop1 = nn.Dropout2d()
            self.fc15 = nn.Linear(1024,128)
            self.drop2 = nn.Dropout2d()
            self.fc16 = nn.Linear(128,10)


        def forward(self,x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(x)
            x = self.bn1(x)
            x = self.relu1(x)


            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pool2(x)
            x = self.bn2(x)
            x = self.relu2(x)

            x = self.conv5(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.pool3(x)
            x = self.bn3(x)
            x = self.relu3(x)

            x = self.conv8(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.pool4(x)
            x = self.bn4(x)
            x = self.relu4(x)

            x = self.conv11(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = self.pool5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            # print(" x shape ",x.size())
            x = x.view(-1,512*4*4)
            x = F.relu(self.fc14(x))
            x = self.drop1(x)
            x = F.relu(self.fc15(x))
            x = self.drop2(x)
            x = self.fc16(x)

            return x

    model = MyNet().to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    print(model)

    # %%
    # define cost/loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # %%
    # model.load_state_dict(torch.load("./myfed_normal_save/model18-32-05.pth")["state_dict"])
    dir = os.listdir("./merge_save")
    modelnames = []
    for mt in dir:
        if(mt[0:len("merge_vgg")] == "merge_vgg"):
            modelnames.append(mt)
    # print(modelnames)
    modelset = []
    for mn in modelnames:
        modelset.append([model.load_state_dict(torch.load("./merge_save/"+mn)["state_dict"]),mn])
    # print(dir)
    savepath = "./metric/vr_metric-merge-VGG16_CIFAR="

    # %%
    len(modelset)

    # %%
    for mode, modelname in modelset:
        parm = {}
        for name,parameters in model.named_parameters():
            parm[name]=parameters
            # print(name)
        tensors = []
        model.eval()
        with torch.no_grad():

            t=5
            group1 = []
            group2 = []
            group3 = []
            group4 = []
            group5 = []
            r = random.sample(range(1,batch_size - 1), t)
            # print(r)
            for i in range(0,t):
                

                for X_test, Y_test in testloader:
                    X_test = X_test.to(device)
                    Y_test = Y_test.to(device)
                

                X_single_data = X_test[r[i]]
                Y_single_data = Y_test[r[i]]

                # print(X_single_data)
                
                layerP = model.module.conv1(torch.unsqueeze(X_single_data,dim=0))
                layerP = model.module.conv2(layerP)
                layerP = model.module.pool1(layerP)
                layerP = model.module.bn1(layerP)
                layerP = model.module.relu1(layerP)
                layerP = model.module.conv3(layerP)
                layerP = model.module.conv4(layerP)
                layerP = model.module.pool2(layerP)
                layerP = model.module.bn2(layerP)
                layerP = model.module.relu2(layerP)
                layerP = model.module.conv5(layerP)
                layerP = model.module.conv6(layerP)
                layerP = model.module.conv7(layerP)
                layerExtract1 = torch.nn.MaxPool2d(kernel_size=5, stride=3)(layerP)
                layerExtract1 = layerExtract1.squeeze().reshape(-1,)
                group1.append(layerExtract1.cpu().detach().numpy())
                layerP = model.module.pool3(layerP)
                layerP = model.module.bn3(layerP)
                layerP = model.module.relu3(layerP)
                layerP = model.module.conv8(layerP)
                layerP = model.module.conv9(layerP)
                layerP = model.module.conv10(layerP)
                layerExtract2 = torch.nn.MaxPool2d(kernel_size=5, stride=4)(layerP)
                layerExtract2 = layerExtract2.squeeze().reshape(-1,)
                group2.append(layerExtract2.cpu().detach().numpy())
                layerP = model.module.pool4(layerP)
                layerP = model.module.bn4(layerP)
                layerP = model.module.relu4(layerP)
                layerP = model.module.conv11(layerP)
                layerP = model.module.conv12(layerP)
                layerP = model.module.conv13(layerP)
                layerExtract3 = torch.nn.MaxPool2d(kernel_size=5, stride=5)(layerP)
                layerExtract3 = layerExtract3.squeeze().reshape(-1,)
                group3.append(layerExtract3.cpu().detach().numpy())
                layerP = model.module.pool5(layerP)
                layerP = model.module.bn5(layerP)
                layerP = model.module.relu5(layerP)

                # x = x.view(-1,512*4*4)
                # x = F.relu(self.fc14(x))
                # x = self.drop1(x)
                # x = F.relu(self.fc15(x))
                # x = self.drop2(x)
                # x = self.fc16(x)

                layerP = layerP.view(-1,512*4*4)
                layerP = F.relu(model.module.fc14(layerP))
                group5.append(layerP.reshape(-1,).cpu().detach().numpy())
                layerP = model.module.drop1(layerP)
                layerP = F.relu(model.module.fc15(layerP))

                # print("LayerP at FC15",layerP.shape)
                # layerExtract4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(layerP)
                # layerExtract4 = layerExtract4.squeeze().reshape(-1,)
                group4.append(layerP.reshape(-1,).cpu().detach().numpy())
        
        l = 0
        for tensors in [group1, group2, group3, group4, group5]:
            # print(len(tensors))
            # print(tensors)
            nb = np.linspace(0,0,tensors[0].shape[0])
            print(nb.shape)
            for i in tensors:
                nb += i
            nb /= t

            ns = np.linspace(0,0,tensors[0].shape[0])
            vr = np.linspace(0,0,tensors[0].shape[0] ** 2).reshape(tensors[0].shape[0],tensors[0].shape[0])

            for i in range(0, len(ns)):
                for nps in tensors:
                    ns[i] += nps[i] ** 2
                ns[i] = ns[i] ** 0.5

            with multiprocessing.Pool(12) as pool:
                # apply_async()函数调用有两种方式，第一种是直接挨个写，第二种是指定参数的方式
                # pool.apply_async(function, args=("function A", 7,),
                #                 callback=note_return)  # 第一种调用方法，指定参数**==**
                # # 第二种调用方法，挨个写，由于第三个参数是别的，所以用了指定的方式：callback=
                # pool.apply_async(function, ("function B", 2,), callback=note_return)
                for i in range(0, len(ns)):
                    pool.apply_async(
                        para_calc, (i, ns, nb, tensors), callback=para_get)
                pool.close()
                pool.join()

            for v in VR:
                vr += v
            
            VR = []
            dt = datetime.datetime.now()
            # np.savetxt("./misc/vr_metric"+dt.strftime("%d--%H-%M-%S")+".dat", vp+vp.T, fmt="%1.5f")
            np.savetxt(savepath+modelname+str(l)+".dat", vr+vr.T, fmt="%1.5f")
            print(savepath+modelname+str(l)+".dat")
            l += 1

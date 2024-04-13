# %%
# Lab 10 MNIST and softmax
from turtle import goto
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

import numpy as np
import os
import datetime
import multiprocessing
# torch.set_printoptions(profile="full")
# np.set_printoptions(threshold=np.inf)


import numpy as np
import copy

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

    # %%
    mnist_train = dsets.MNIST(root='MNIST_data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)

    # %%
    class MyNet(torch.nn.Module):
        def __init__(self):
            super(MyNet, self).__init__()
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 25, kernel_size=3),
                torch.nn.BatchNorm2d(25),
                torch.nn.ReLU(inplace=True)
            )
    
            self.layer2 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
    
            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(25, 50, kernel_size=3),
                torch.nn.BatchNorm2d(50),
                torch.nn.ReLU(inplace=True)
            )
    
            self.layer4 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
    
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(50 * 5 * 5, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, 1024),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(1024, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, 10)
            )
    
        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = MyNet().to(device)

    print(model)

    # %%
    # define cost/loss & optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)    # Softmax is internally computed.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # %%
    # model.load_state_dict(torch.load("./myfed_normal_save/model18-32-05.pth")["state_dict"])
    dir = os.listdir("./merge_save")
    modelnames = []
    for mt in dir:
        if(mt[0:len("merge_mnist_moreFC_")] == "merge_mnist_moreFC_"):
            modelnames.append(mt)
    # print(modelnames)
    modelset = []
    for mn in modelnames:
        modelset.append([model.load_state_dict(torch.load("./merge_save/"+mn)["state_dict"]),mn])
    # print(dir)
    savepath = "./metric/vr_metric-mnist-more_layers="

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
            r = random.sample(range(1,batch_size - 1), t)
            # print(r)
            for i in range(0,t):
                

                for X_test, Y_test in test_loader:
                    X_test = X_test.to(device)
                    Y_test = Y_test.to(device)
                

                X_single_data = X_test[r[i]]
                Y_single_data = Y_test[r[i]]

                # print(X_single_data)
                
                layerP = model.layer1(torch.unsqueeze(X_single_data,dim=0))
                layerP = model.layer2(layerP)
                layerExtract1 = torch.nn.MaxPool2d(kernel_size=4, stride=2)(layerP)
                layerExtract1 = layerExtract1.squeeze().reshape(-1,)
                # print(layerExtract1.cpu().detach().numpy())
                group1.append(layerExtract1.cpu().detach().numpy())
                layerP = model.layer3(layerP)
                layerP = model.layer4(layerP)
                layerExtract2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(layerP)
                layerExtract2 = layerExtract2.squeeze().reshape(-1,)
                # print(layerExtract2.shape)
                group2.append(layerExtract2.cpu().detach().numpy())
                layerP = layerP.view(layerP.size(0), -1)
                # print(layerP.shape)
                # layerP = model.fc(layerP)
                tensor1 = torch.mm(layerP, parm['fc.0.weight'].data.permute(1,0)) + parm['fc.0.bias']
                # print(tensor1.shape)
                tensor2 = torch.mm(tensor1, parm['fc.2.weight'].data.permute(1,0)) + parm['fc.2.bias']
                layerExtract3 = tensor2.squeeze().reshape(-1,)
                # print(layerExtract3.shape)
                group3.append(layerExtract3.cpu().detach().numpy())
                tensor3 = torch.mm(tensor2, parm['fc.4.weight'].data.permute(1,0)) + parm['fc.4.bias']
                layerExtract4 = tensor3.squeeze().reshape(-1,)
                # print(layerExtract4.shape)
                group4.append(layerExtract4.cpu().detach().numpy())
                # tensor4 = torch.mm(tensor3, parm['fc.6.weight'].data.permute(1,0)) + parm['fc.6.bias']
                
                # layerExtract2 = layerP.squeeze().reshape(-1,128)
                # # print(layerExtract2.shape)
                # group3.append(layerExtract2.cpu().detach().numpy()[0])
                # layerP = layerP.detach().squeeze().reshape(-1,512*4*4)

                # tensor1 = torch.mm(layerP, parm['module.fc14.weight'].data.permute(1,0)) + parm['module.fc14.bias']
                # tensor1 = torch.mm(tensor1, parm['module.fc15.weight'].data.permute(1,0)) + parm['module.fc15.bias']
                
                # group4.append(tensor1.cpu().detach().numpy()[0])

        l = 0
        for tensors in [group1, group2, group3, group4]:
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

            with multiprocessing.Pool(10) as pool:
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

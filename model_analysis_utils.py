import torch
import torch.nn.functional as F
import random

def getLayer(layerP, kernel_size=5, stride=4):
    layerExtract1 = torch.nn.MaxPool2d(
                kernel_size=kernel_size, stride=stride)(layerP)
    layerExtract1 = layerExtract1.squeeze().reshape(-1,)
    return layerExtract1.cpu().detach().numpy()

def evaulation_vgg(model, test_loader, t=10, batch_size=128, device="cuda"):
    model.eval()
    with torch.no_grad():

        group1 = []
        group2 = []
        group3 = []
        group4 = []
        group5 = []
        r = random.sample(range(1, batch_size - 1), t)

        for X, Y in test_loader:
            X_test = X.to(device)
#             Y_test = Y_test.to(device)
            break
        for i in range(0, t):

            X_single_data = X_test[r[i]]

            layerP = model.conv1(torch.unsqueeze(X_single_data, dim=0))
            layerP = model.conv2(layerP)
            layerP = model.pool1(layerP)
            layerP = model.bn1(layerP)
            layerP = model.relu1(layerP)
            layerP = model.conv3(layerP)
            layerP = model.conv4(layerP)
            layerP = model.pool2(layerP)
            layerP = model.bn2(layerP)
            layerP = model.relu2(layerP)
            layerP = model.conv5(layerP)
            layerP = model.conv6(layerP)
            layerP = model.conv7(layerP)
            layerExtract1 = torch.nn.MaxPool2d(
                kernel_size=5, stride=3)(layerP)
            layerExtract1 = layerExtract1.squeeze().reshape(-1,)
            group1.append(layerExtract1.cpu().detach().numpy())
            layerP = model.pool3(layerP)
            layerP = model.bn3(layerP)
            layerP = model.relu3(layerP)
            layerP = model.conv8(layerP)
            layerP = model.conv9(layerP)
            layerP = model.conv10(layerP)
            layerExtract2 = torch.nn.MaxPool2d(
                kernel_size=5, stride=4)(layerP)
            layerExtract2 = layerExtract2.squeeze().reshape(-1,)
            group2.append(layerExtract2.cpu().detach().numpy())
            layerP = model.pool4(layerP)
            layerP = model.bn4(layerP)
            layerP = model.relu4(layerP)
            layerP = model.conv11(layerP)
            layerP = model.conv12(layerP)
            layerP = model.conv13(layerP)
            layerExtract3 = torch.nn.MaxPool2d(
                kernel_size=5, stride=5)(layerP)
            layerExtract3 = layerExtract3.squeeze().reshape(-1,)
            group3.append(layerExtract3.cpu().detach().numpy())
            layerP = model.pool5(layerP)
            layerP = model.bn5(layerP)
            layerP = model.relu5(layerP)
            layerP = layerP.view(-1, 512*4*4)
            layerP = F.relu(model.fc14(layerP))
            group5.append(layerP.reshape(-1,).cpu().detach().numpy())
            layerP = model.drop1(layerP)
            layerP = F.relu(model.fc15(layerP))
            group4.append(layerP.reshape(-1,).cpu().detach().numpy())
        return [group1, group2, group3, group4, group5]


def evaulation_lenet(model, test_loader, t=10, batch_size=128, device="cuda"):
    model.eval()
    parm = {}
    for name,parameters in model.named_parameters():
        parm[name]=parameters
    with torch.no_grad():

        group1 = []
        group2 = []
        group3 = []
        group4 = []
        r = random.sample(range(1, batch_size - 1), t)
#         print(r)
        for i in range(0, t):
            print(r[i],end=",")
            for X_test, Y_test in test_loader:
                X_test = X_test.to(device)
#                 Y_test = Y_test.to(device)
                break

            X_single_data = X_test[r[i]]

            layerP = model.layer1(torch.unsqueeze(X_single_data, dim=0))
            layerP = model.layer2(layerP)
            layerExtract1 = torch.nn.MaxPool2d(kernel_size=4, stride=2)(layerP)
            layerExtract1 = layerExtract1.squeeze().reshape(-1,)
            group1.append(layerExtract1.cpu().detach().numpy())
            layerP = model.layer3(layerP)
            layerP = model.layer4(layerP)
            layerExtract2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(layerP)
            layerExtract2 = layerExtract2.squeeze().reshape(-1,)
            group2.append(layerExtract2.cpu().detach().numpy())
            layerP = layerP.view(layerP.size(0), -1)
            tensor1 = torch.mm(layerP, parm['fc.0.weight'].data.permute(
                1, 0)) + parm['fc.0.bias']
            tensor2 = torch.mm(tensor1, parm['fc.2.weight'].data.permute(
                1, 0)) + parm['fc.2.bias']
            layerExtract3 = tensor2.squeeze().reshape(-1,)
            group3.append(layerExtract3.cpu().detach().numpy())
            tensor3 = torch.mm(tensor2, parm['fc.4.weight'].data.permute(
                1, 0)) + parm['fc.4.bias']
            layerExtract4 = tensor3.squeeze().reshape(-1,)
            group4.append(layerExtract4.cpu().detach().numpy())
        return [group1, group2, group3, group4]

def evaulation_mnist(model, test_loader, t=10, batch_size=64, device="cuda"):
    model.eval()
    parm = {}
    for name,parameters in model.named_parameters():
        parm[name]=parameters
    with torch.no_grad():

        group1 = []
        group2 = []
        group3 = []
        group4 = []
        r = random.sample(range(1, batch_size - 1), t)
#         print(r)
        for i in range(0, t):
            print(r[i],end="")
            for X_test, Y_test in test_loader:
                X_test = X_test.to(device)
#                 Y_test = Y_test.to(device)
                break

            X_single_data = X_test[r[i]]

            layerP = F.relu(model.conv1(torch.unsqueeze(X_single_data, dim=0)))
            layerExtract1 = torch.nn.MaxPool2d(kernel_size=5, stride=4)(layerP)
            layerExtract1 = layerExtract1.squeeze().reshape(-1,)
            group1.append(layerExtract1.cpu().detach().numpy())
            layerP = F.max_pool2d(layerP, 2, 2)
            layerP = F.relu(model.conv2(layerP))
            layerExtract2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(layerP)
            layerExtract2 = layerExtract2.squeeze().reshape(-1,)
            group2.append(layerExtract2.cpu().detach().numpy())
            layerP = F.max_pool2d(layerP, 2, 2)
            layerP = layerP.view(-1, 4 * 4 * 50)
            layerP = F.relu(model.fc1(layerP))
            layerExtract3 = layerP.squeeze().reshape(-1,)
            group3.append(layerExtract3.cpu().detach().numpy())
            layerP = model.fc2(layerP)
            layerExtract4 = layerP.squeeze().reshape(-1,)
            group4.append(layerExtract4.cpu().detach().numpy())


            # layerP = model.layer3(layerP)
            # layerP = model.layer4(layerP)

            # layerP = layerP.view(layerP.size(0), -1)
            # tensor1 = torch.mm(layerP, parm['fc.0.weight'].data.permute(
            #     1, 0)) + parm['fc.0.bias']
            # tensor2 = torch.mm(tensor1, parm['fc.2.weight'].data.permute(
            #     1, 0)) + parm['fc.2.bias']
            
            # tensor3 = torch.mm(tensor2, parm['fc.4.weight'].data.permute(
            #     1, 0)) + parm['fc.4.bias']
            
        return [group1, group2, group3, group4]


def evaulation_lenet_tail(model, test_loader, t=10, batch_size=128, device="cuda"):
    model.eval()
    parm = {}
    for name,parameters in model.named_parameters():
        parm[name]=parameters
    with torch.no_grad():

        group1 = []
        group2 = []
        group3 = []
        group4 = []
        r = random.sample(range(1, batch_size - 1), t)
#         print(r)
        for i in range(0, t):
            print(r[i],end=",")
            for X_test, Y_test in test_loader:
                X_test = X_test.to(device)
#                 Y_test = Y_test.to(device)
                break

            X_single_data = X_test[r[i]]

            layerP = F.relu(model.conv1((torch.unsqueeze(X_single_data, dim=0))))
            layerP = F.max_pool2d(layerP, 2)
            layerExtract1 = torch.nn.MaxPool2d(kernel_size=4, stride=2)(layerP)
            layerExtract1 = layerExtract1.squeeze().reshape(-1,)
            group1.append(layerExtract1.cpu().detach().numpy())
            layerP = F.relu(model.conv2(layerP))
            layerP = F.max_pool2d(layerP, 2)
            layerExtract2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)(layerP)
            layerExtract2 = layerExtract2.squeeze().reshape(-1,)
            group2.append(layerExtract2.cpu().detach().numpy())
            layerP = layerP.view(layerP.size(0), -1)
            tensor1 = F.relu(model.fc1(layerP))
            # tensor1 = torch.mm(layerP, parm['fc.0.weight'].data.permute(
            #     1, 0)) + parm['fc.0.bias']
            tensor2 = F.relu(model.fc2(tensor1))
            # tensor2 = torch.mm(tensor1, parm['fc.2.weight'].data.permute(
            #     1, 0)) + parm['fc.2.bias']
            layerExtract3 = tensor2.squeeze().reshape(-1,)
            group3.append(layerExtract3.cpu().detach().numpy())
            # tensor3 = torch.mm(tensor2, parm['fc.4.weight'].data.permute(
            #     1, 0)) + parm['fc.4.bias']
            tensor3 = F.relu(model.fc2(tensor2))
            layerExtract4 = tensor3.squeeze().reshape(-1,)
            group4.append(layerExtract4.cpu().detach().numpy())
        return [group1, group2, group3, group4]

def evaulation_resnet(model, test_loader, t=10, batch_size=128, device="cuda"):
    model.eval()
    parm = {}
    for name,parameters in model.named_parameters():
        parm[name]=parameters
    with torch.no_grad():

        group1 = []
        group2 = []
        group3 = []
        group4 = []
        r = random.sample(range(1, batch_size - 1), t)
#         print(r)
        for i in range(0, t):
            print(r[i],end=",")
            for X_test, Y_test in test_loader:
                X_test = X_test.to(device)
#                 Y_test = Y_test.to(device)
                break

            X_single_data = X_test[r[i]]


            layerP = F.relu(model.bn1(model.conv1(torch.unsqueeze(X_single_data, dim=0))))
            group4.append(getLayer(layerP))
            layerP = model.layer1(layerP)
            group1.append(getLayer(layerP))
            layerP = model.layer2(layerP)
            group2.append(getLayer(layerP))
            layerP = model.layer3(layerP)
            group3.append(getLayer(layerP))
            layerP = F.avg_pool2d(layerP, layerP.size()[3])
            layerP = layerP.view(layerP.size(0), -1)
            layerP = model.linear(layerP)


        return [group1, group2, group3, group4]
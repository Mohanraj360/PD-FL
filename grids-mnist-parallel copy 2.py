# %% [markdown]
# [Persim 0.3.1 documentation](https://persim.scikit-tda.org/en/latest/notebooks/Persistence%20images.html#Generate-a-persistence-diagram-using-Ripser)

# %%
from itertools import product
import multiprocessing

import time
from unicodedata import name
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from ripser import Rips
from persim import PersistenceImager

import os

import seaborn
import pandas as pd

import copy
from memory_profiler import profile
import psutil
import traceback

# torch.set_printoptions(profile="full")
# np.set_printoptions(threshold=np.inf)


# %%
dgms = []
H0_dgm = []
H1_dgm = []
# %%
def gridsize(h_gdm):
    s = [999,0,999,0]
    for h in h_gdm:
        s[0] = s[0] if h[0] > s[0] else h[0]
        s[1] = s[1] if h[0] < s[1] else h[0]
        s[2] = s[2] if h[1] > s[2] else h[1]
        s[3] = s[3] if h[1] < s[3] else h[1]
    return s
def makegrid(h_gdm, s):
    height = (s[1] - s[0]) / 126
    weight = (s[3] - s[2]) / 126
    grid = np.zeros((128,128))
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


def add_rips(d,idx):
    global dgms
    global H0_dgm
    global H1_dgm
    if idx % 10 == 0:
        print(idx)
    rips = Rips(maxdim=2)
    print("calculating",end="")
    for i in range(999):
        if psutil.virtual_memory().percent < 60:
            break
        if i % 10 == 0 :
            print("'sleeping {}'".format(i), end="")
        time.sleep(1)
    a = rips.fit_transform(d[0], distance_matrix=True)
    print("'calculated with a.len={}'".format(len(a[1])),end="")
    value = (a,d[1])
    dgms.append(value)
    H0_dgm.append((value[0][0], value[1]))
    H1_dgm.append((value[0][1], value[1]))
    print("added")
    print("len(H1_dgm) = {}".format(len(H1_dgm)))
    return value

def get_rips(value):
    global dgms
    global H0_dgm
    global H1_dgm
    dgms.append(value)
    H0_dgm.append((value[0][0], value[1]))
    H1_dgm.append((value[0][1], value[1]))
    # print("added")
    # print("len(H1_dgm) = {}".format(len(H1_dgm)))

def parallel(d,idx):
    global dgms
    global H0_dgm
    global H1_dgm
    if idx % 10 == 0:
        print(idx)
    rips = Rips(maxdim=1)
    print("calculating",end="")
    for i in range(999):
        if psutil.virtual_memory().percent < 60:
            break
        if i % 10 == 0 :
            print("'sleeping {}'".format(i), end="")
        time.sleep(1)
    a = rips.fit_transform(d[0], distance_matrix=True)
    print("'calculated with a.len={}'".format(len(a[1])),end="")
    value = (a,d[1])
    return value

def errorcallback(args):
    try:
        print("ERROR0:", args)
    except:
        print("ERROR1:", traceback.format_exc(3))

if __name__ == "__main__":
    dir = os.listdir("./metric")
    data = []
    for metric in dir:
        if metric[len("vr_metric-single-mnist_moreFC="):len("vr_metric-single-mnist_moreFC=")+len("mnist_moreFC")] == "mnist_moreFC":
            data.append([np.loadtxt("./metric/"+metric), metric])
            # print(metric)
        # if len(data) > 80:
        #     break
    print(len(data))


    for i in range(len(data)):
        data[i][0] = abs(data[i][0]) * 100
        test = [j for j in data[i][0][0:-1] if j.any() != 0]
        test = np.array(test)
        # print(test.shape)
        test = [j for j in test.T[:,0:-1] if j.any() != 0]
        # grid = pd.DataFrame(test)
        # plot = seaborn.heatmap(grid)
        # plt.title("data")
        # plt.show()
        test = np.array(test)
        if test.shape[0] > test.shape[1]:
            test = test[0:test.shape[1],:]
        elif test.shape[0] < test.shape[1]:
            test = test[:,0:test.shape[0]]
        data[i][0] = np.array(test)

    print("multiprocessing")
    multiprocessing.freeze_support()
    for top in range(0,len(data)-1,80):
        print(top)
        with multiprocessing.Pool(8) as pool:
            for idx, d in enumerate(data[top:min(top+80,len(data))]):
                pool.apply_async(
                        parallel, (d,idx,),callback=get_rips, error_callback=errorcallback)
                # print(1)
                
            pool.close()
            pool.join()
                # a = dgms[-1]
                # H1_dgm.append([a[0][1], a[1]])
                # H0_dgm.append([a[0][0], a[1]])

        print("len(H1_dgm) = {}".format(len(H1_dgm)))

        # %%
        for h1 in H1_dgm:
            # print(len(h1[0]))
            # print(gridsize(h1[0]))
            # print(makegrid(h1[0],gridsize(h1[0])))
            np.savetxt("./grids/h_"+h1[1], makegrid(h1[0],gridsize(h1[0])))
            print("./grids/h_"+h1[1])
            # grid = pd.DataFrame(makegrid(h1[0],gridsize(h1[0])))
            # plot = seaborn.heatmap(grid)
            # plt.title(h1[1])
            # plt.show()
        
        dgms = []
        H0_dgm = []
        H1_dgm = []


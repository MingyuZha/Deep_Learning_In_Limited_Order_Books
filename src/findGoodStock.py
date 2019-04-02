import torch
import numpy as np
np.set_printoptions(suppress=True)
import h5py
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset

def _generateDateList(root, stock_name, train=True):
    dataFile = h5py.File(root+stock_name+".hdf5")
    dates_path = "./dates_file/"
    if (not os.path.exists(dates_path)):
        os.mkdir(dates_path)
    if (os.path.isfile("./dates_file/"+stock_name+"_dates.npy")):
        dates = np.load("./dates_file/"+stock_name+"_dates.npy")
    else:
        dates = []
        for key in dataFile.keys():
            if (key.startswith("T")):
                dates.append(key[2 : ])
        np.save("./dates_file/"+stock_name+"_dates.npy", dates)
    if (train):
        start_date = "2014-01-01"
        end_date = "2014-01-15"
    else:
        start_date = "2017-01-01"
        end_date = "2017-01-05"
    date_list = np.array((pd.date_range(start=start_date, end=end_date)).date)
    date_list = [item.strftime('%Y-%m-%d')for item in date_list]
    date_list = sorted(list(set(date_list)&set(dates)))
    dataFile.close()
    return date_list

def _generateLabels(root, stock_name):
    dataset = h5py.File(root+"/"+stock_name+".hdf5", "r")
    date_list = _generateDateList(root, stock_name, True)
    Labels = []
    for date in date_list:
        X_date = "X_" + date
        Y_date = "Y_" + date
        X = dataset[X_date][0:-1]
        Y = dataset[Y_date]
        I = ((X <= -9999999999) | (X >= 9999999999))
        X[I] = 0.
        X_ask_best = copy.deepcopy(X[:, 0])
        next_best_ask = copy.deepcopy(Y[:, 0])
        magnitude = (next_best_ask - X_ask_best) / 100
        I = (magnitude > 4)
        magnitude[I] = 4
        I = (magnitude < -4)
        magnitude[I] = -4
        Y_new = magnitude
        Y_new += 4

        Labels.append(Y_new)
    Labels = np.hstack(Labels)
    dataset.close()
    return Labels

if __name__ == "__main__":
    stock_name_list = []

    stock_name = "AAPL"
    root = "/projects/sciteam/bahp/OneSecondData1000/"

    Labels = _generateLabels(root, stock_name)
    Labels = Labels[Labels != 4]
    I = (Labels > 4)
    Labels[I] -= 1
    # plt.hist(Labels, bins=[0,1,2,3,4,5,6,7,8], normed=1)
    # plt.show()
    hist = np.histogram(Labels, bins=[0,1,2,3,4,5,6,7,8], normed=1)
    print (hist)



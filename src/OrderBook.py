import torch
import numpy as np
import h5py
import pandas as pd
import copy
import os
from torch.utils.data.dataset import Dataset


def _dataClean(raw_data, Y):
    cleanedOrderBook = []
    cleanedNextPrice = []
    for index, data in enumerate(raw_data):
        if (index == raw_data.shape[0]-1): continue
        cur_mid = (data[0] + data[2]) * .5
        next_mid = (Y[index, 0] + Y[index, 1])*.5
        if (cur_mid == next_mid): continue
        cleanedOrderBook.append(data)
        cleanedNextPrice.append(Y[index])
    return np.asarray(cleanedOrderBook), np.asarray(cleanedNextPrice)

class LimitOrderBook(Dataset):
    def __init__(self, root, stock_name, train=True, num_levels=10, num_inputs=20, sequence_len=20):
        """
        root: "./xxx"
        stock_name: "AAPL"
        date: "2014-01-05"
        """
        print ("Start building the dataset...")
        self.root = root
        self.sequence_len = sequence_len
        self.dataset = h5py.File(self.root + "/" + stock_name + ".hdf5", "r")
        if (train):
            date_list = _generateDateList(root, stock_name, True)
        else:
            date_list = _generateDateList(root, stock_name, False)
        print ("Generated date list...")
        self.X_train = []
        self.Y_train = []
        for date in date_list:
            X_date = "X_"+date
            Y_date = "Y_"+date
            X = self.dataset[X_date][0:-1]
            Y = self.dataset[Y_date]
            I = ((X <= -9999999999) | (X >= 9999999999))
            X[I] = 0.
            X_ask_best = copy.deepcopy(X[:, 0])
            X_bid_best = copy.deepcopy(X[:, 2])
            X_new = np.zeros((X.shape[0], num_inputs))

            ##Fixed level, put each level's size into the X_new, if not existed, keep it 0
            for level in range(num_levels):
                for col in range(0, 4*num_levels, 4):
                    I_ask = ((X[:, col] - X_ask_best) / 100 == level)
                    I_bid = ((X_bid_best - X[:, col+2]) / 100 == level)
                    X_new[I_ask, level] = X[I_ask, col+1] / 10000.
                    X_new[I_bid, level+num_levels] = X[I_bid, col+3] / 10000.

            mid_price = (X_ask_best + X_bid_best) / 2.
            Y = (Y[:, 0] + Y[:, 1]) / 2.
            Y = Y - mid_price
            Y[Y > 0] = 1
            Y[Y < 0] = -1
            Y += 1

            self.X_train.append(X_new)
            self.Y_train.append(Y)
        print ("Generated training data...")
        self.X_train = np.vstack(self.X_train)
        self.Y_train = np.hstack(self.Y_train)
        self.size = len(self.X_train)-sequence_len+1
        self.dataset.close()

    def __getitem__(self, index):
        """
        x: [sequence_len, 20]
        price_move: [sequence_len, 1]
        """
        # x = self.order_book[index : index+self.sequence_len]
        # cur_mid_price = (x[:, 0] + x[:, 2]) / 2.0
        # x = x[:,1::2] #Only keep the even columns
        # next_mid_price = np.mean(self.next_price[index: index+self.sequence_len], axis=1)
        # price_moveup = np.float32(cur_mid_price < next_mid_price).reshape(-1,1)
        # x, price_moveup = torch.from_numpy(x), torch.from_numpy(price_moveup)
        # price_moveup = torch.zeros(len(x), 2).scatter_(1, price_moveup, 1)
        # next_mid_price = torch.from_numpy(next_mid_price).view(-1,1)
        x = self.X_train[index:(index+self.sequence_len)]
        y = self.Y_train[index:(index+self.sequence_len)]
        return x, y

    def __len__(self):
        # length = len(self.next_price) - self.sequence_len - 2
        return self.size


class LimitOrderBook_magnitude(Dataset):
    def __init__(self, root, stock_name, mode="classification", train=True, num_levels=10, num_inputs=20, sequence_len=20):
        """
        root: "./xxx"
        stock_name: "AAPL"
        date: "2014-01-05"
        mode: options:[classification, regression]
        """
        print ("Start building the dataset...")
        self.root = root
        self.sequence_len = sequence_len
        self.dataset = h5py.File(self.root + "/" + stock_name + ".hdf5", "r")
        if (train):
            date_list = _generateDateList(root, stock_name, True)
        else:
            date_list = _generateDateList(root, stock_name, False)
        print ("Generated date list...")
        self.X_train = []
        self.Y_train = []
        for date in date_list:
            X_date = "X_"+date
            Y_date = "Y_"+date
            X = self.dataset[X_date][0:-1]
            Y = self.dataset[Y_date]
            I = ((X <= -9999999999) | (X >= 9999999999))
            X[I] = 0.
            X_ask_best = copy.deepcopy(X[:, 0])
            X_bid_best = copy.deepcopy(X[:, 2])
            X_new = np.zeros((X.shape[0], num_inputs))

            ##Fixed level, put each level's size into the X_new, if not existed, keep it 0
            for level in range(num_levels):
                for col in range(0, 4*num_levels, 4):
                    I_ask = ((X[:, col] - X_ask_best) / 100 == level)
                    I_bid = ((X_bid_best - X[:, col+2]) / 100 == level)
                    X_new[I_ask, level] = X[I_ask, col+1] / 10000.
                    X_new[I_bid, level+num_levels] = X[I_bid, col+3] / 10000.

            next_best_ask = copy.deepcopy(Y[:, 0])
            if (mode == "classification"):
                magnitude = (next_best_ask - X_ask_best) / 100
                Y_new = copy.deepcopy(magnitude)
                I = (magnitude >= 4)
                Y_new[I] = 4
                Y_new[magnitude <= -4] = -4
                Y_new += 4
            else:
                magnitude = (next_best_ask - X_ask_best)/100.
                Y_new = copy.deepcopy(magnitude)

            self.X_train.append(X_new)
            self.Y_train.append(Y_new)
        if (train):
            print ("Generated training data...")
        else :
            print("Generated testing data...")
        self.X_train = np.vstack(self.X_train)
        self.Y_train = np.hstack(self.Y_train)
        self.size = int(len(self.X_train) / sequence_len)
        self.dataset.close()

    def __getitem__(self, index):
        start_index = index * self.sequence_len
        x = self.X_train[start_index : (start_index+self.sequence_len)]
        y = self.Y_train[start_index : (start_index+self.sequence_len)]
        return x, y

    def __len__(self):
        return self.size



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
        end_date = "2016-12-31"
    else:
        start_date = "2017-01-01"
        end_date = "2017-03-31"
    date_list = np.array((pd.date_range(start=start_date, end=end_date)).date)
    date_list = [item.strftime('%Y-%m-%d')for item in date_list]
    date_list = sorted(list(set(date_list)&set(dates)))
    dataFile.close()
    return date_list



if __name__ == "__main__":
    dataset = LimitOrderBook_magnitude("../../", "AAPL", mode="regression")
    dataLoader = torch.utils.data.DataLoader(dataset, 128, False, num_workers=0)
    for index, data in enumerate(dataLoader):
        x, y = data
        y = y.numpy()
        print (y)
        break
    # print ("Total samples: ", np.sum(counts))
    # print("Class Distribution: ", counts







"""
the following is taken and modified from: https://github.com/RuwanT/merck/blob/master/data_preprocessing.py
convert the merck data-set suitable to be fead to the CNN

1) remove columns that does not appear in both training and test
2) normalize the activation to have zero mean and 1 std (z-score)
3) rescale the features to 0-1 by dividing each column by its training max or y = log(x+1)

"""

import pandas as pd
import torch
import pickle
import numpy as np
# from nutsflow import *
# from nutsml import *
import sys
import os

data_root = './data/Merck/'
save_root = './data/Merck/preprocessed/'
os.makedirs(save_root, exist_ok=True)
FEATURE_SCALE = 'log'   # 'uniform'


dataset_names = ['3A4', 'CB1', 'DPP4', 'HIVINT', 'HIVPROT', 'LOGD', 'METAB', 'NK1', 'OX1', 'OX2', 'PGP', 'PPB', 'RAT_F', 'TDI', 'THROMBIN']

stat_hold = list()  # hold the mean and standard deviation for each data-set

for dataset_name in dataset_names:

    test_filename = data_root + dataset_name + '_test_disguised.csv'
    train_filename = data_root + dataset_name + '_training_disguised.csv'

    test_filename_save = save_root + dataset_name + '_test_disguised.csv'
    train_filename_save = save_root + dataset_name + '_training_disguised.csv'

    print('Preprocessing dataset ', dataset_name)

    train = pd.read_csv(train_filename)
    test = pd.read_csv(test_filename)

    print(len(train.columns.values))
    print(len(test.columns.values))

    train_inx_set = set(train.columns.values)
    test_inx_set = set(test.columns.values)

    # remove columns that are not common to both training and test sets
    train_inx = [inx for inx in train.columns.values if inx in set.intersection(train_inx_set, test_inx_set)]
    test_inx = [inx for inx in test.columns.values if inx in set.intersection(train_inx_set, test_inx_set)]

    train = train[train_inx]
    test = test[test_inx]

    train.iloc[:, 2:] = train.iloc[:, 2:].apply(pd.to_numeric)
    train.iloc[:, 2:] = train.iloc[:, 2:].astype(np.float64)
    test.iloc[:, 2:] = test.iloc[:, 2:].apply(pd.to_numeric)

    print(train.shape)
    print(test.shape)

    # Normalize activations
    X = np.asarray(train.Act)
    x_mean = np.mean(X)
    x_std = np.std(X)

    stat_hold.append((dataset_name, x_mean, x_std, train.shape, test.shape))

    train.Act = (train.Act - x_mean) / x_std
    test.Act = (test.Act - x_mean) / x_std

    # rescale features
    if FEATURE_SCALE == 'log':
        train.iloc[:, 2:] = np.log(train.iloc[:, 2:] + 1)
        test.iloc[:, 2:] = np.log(test.iloc[:, 2:] + 1)

    elif FEATURE_SCALE == 'uniform':
        max_feature = train.max(axis=0)[2:]
        train.iloc[:, 2:] = train.iloc[:, 2:] / max_feature
        test.iloc[:, 2:] = test.iloc[:, 2:] / max_feature
    else:
        sys.exit("Feature normalization method not defined correctly, check FEATURE_SCALE. ")

    train_names = train.iloc[:, 0].values.tolist()
    test_names = test.iloc[:, 0].values.tolist()

    train_data = train.iloc[:, 1:].to_numpy()
    test_data = test.iloc[:, 1:].to_numpy()

    with open(f"{save_root}/{dataset_name}_train_names.txt", "w") as f:
        for i, line in enumerate(train_names):
            end = "\n" if i == len(train_names) - 1 else ""
            f.write(f"{line}{end}")

    with open(f"{save_root}/{dataset_name}_test_names.txt", "w") as f:
        for i, line in enumerate(test_names):
            end = "\n" if i == len(test_names) - 1 else ""
            f.write(f"{line}{end}")

    torch.save(torch.from_numpy(train_data), f"{save_root}/{dataset_name}_train.pt")
    torch.save(torch.from_numpy(test_data), f"{save_root}/{dataset_name}_test.pt")

    # save data to csv
    # train.to_csv(train_filename_save, index=False)
    # test.to_csv(test_filename_save, index=False)

    print('Done dataset ', dataset_name)
    for t in stat_hold:
        print(t)

with open(f"{save_root}/stats.pkl", "wb") as f:
    pickle.dump(stat_hold, f)

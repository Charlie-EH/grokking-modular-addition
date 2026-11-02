"""
This file contains code to build the relevant grokking dataset for addition modulo P split - as per the power paper
- randomly into train and test sets including dataloaders and batch size for complete definition of the dataset
"""

import torch
import numpy as np


def make_data(p, train_frac):
    '''Return (train, test) tensors of shape (N, 3); columns are a, b, (a + b) % p'''

    data = []
    for i in range(p):
        for j in range(p):
            data.append([i,j,(i+j)%p]) # structured such that i and j are input data and this column is label (the result of the sum)
    data = np.array(data)

    # split data
    np.random.shuffle(data)
    train_data = data[:int(len(data)*train_frac)]
    test_data = data[int(len(data)*train_frac):]
    train_data = torch.tensor(train_data,dtype=torch.long)
    test_data = torch.tensor(test_data,dtype=torch.long)
    
    return  train_data, test_data


# below ensures data generation is correct

if __name__ == "__main__": # runs this code only when file is called directly
    p, train_frac= 53,0.5
    train,test = make_data(p,train_frac)

    # 1. Shapes
    n_train = int(p * p * train_frac)
    assert train.shape == (n_train, 3), train.shape
    assert test.shape == (p * p - n_train, 3), test.shape

    # 2. Labels are right
    for split in (train, test):
        assert ((split[:, 0] + split[:, 1]) % p == split[:, 2]).all()

    # 3. No overlap, nothing missing
    train_ids = set((train[:, 0] * p + train[:, 1]).tolist())
    test_ids = set((test[:, 0] * p + test[:, 1]).tolist())
    assert len(train_ids & test_ids) == 0          # no pair in both
    assert len(train_ids | test_ids) == p * p      # every pair present

    print("All checks passed")
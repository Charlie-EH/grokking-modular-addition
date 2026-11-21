"""
This file contains code to build the relevant grokking dataset for addition modulo P split - as per the power paper
- randomly into train and test sets for complete definition of the dataset
"""

import torch
import numpy as np


def make_data(p, train_frac, eq_token=False):
    '''Return (train, test) tensors of shape [a, b, label] or [a, b, p, labl] wiht eq_token=True'''

    data = []
    for i in range(p):
        for j in range(p):
            if eq_token==False:
                data.append([i,j,(i+j)%p]) # structured such that i and j are input data and this column is label (the result of the sum)
            else:
                data.append([i,j,p,(i+j)%p])
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
    
    def check(p,train_frac,eq_token):


        train,test = make_data(p,train_frac,eq_token)

        n_cols = 4 if eq_token == True else 3

        # 1. Shapes
        n_train = int(p * p * train_frac)
        assert train.shape == (n_train, n_cols), train.shape
        assert test.shape == (p * p - n_train, n_cols), test.shape

        # 2. Labels are right
        for split in (train, test):
            assert ((split[:, 0] + split[:, 1]) % p == split[:, -1]).all()

        # 3. column 2 == p i.e check the "=" is in vocab
        if eq_token == True:
            for split in (train,test):
                assert (split[:,2] == p).all()

        # 4. No overlap, nothing missing
        train_ids = set((train[:, 0] * p + train[:, 1]).tolist())
        test_ids = set((test[:, 0] * p + test[:, 1]).tolist())
        assert len(train_ids & test_ids) == 0          # no pair in both
        assert len(train_ids | test_ids) == p * p      # every pair present
        print(f"ok: p={p}, eq_token={eq_token}, train={len(train)}, test={len(test)}")

    np.random.seed(0)
    check(53,0.5,eq_token=False) #tests MLP on base parameters
    check(113,0.3,eq_token=True) #tests transformer on base settings
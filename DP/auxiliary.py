import numpy as np
import pandas as pd


def is_two_power(n):
    if n == 1:
        return True
    res = 1
    while res < n:
        res = res << 1
    if res == n:
        return True
    else:
        return False

def sparse_data(data, frac=1, sparse_ratio=100):
    empty_data = pd.DataFrame(columns=data.columns, index=[i for i in range(sparse_ratio * len(data))])
    shuffle_data = pd.concat([data, empty_data]).sample(frac=frac).reset_index(drop=True)
    return shuffle_data


def insert_deletion_data(data, concentrate=True):
    # We
    if concentrate:
        data.insert(data.shape[1], 'update', np.nan)
        for i in range(len(data)):
            if data.loc[i].all():
                continue
            else:
                data.loc[i, 'update'] = 1
                data.loc[len(data)] = data.loc[i]
                data.loc[len(data) - 1, 'update'] = -1
    else:
        data.insert(data.shape[1], 'update', np.nan)
        for i in reversed(range(len(data))):
            if data.loc[i].all():
                continue
            else:
                data.loc[i, 'update'] = 1
                row_index = np.random.randint(i, len(data))
                deletion_row = data.loc[i:i].copy()
                deletion_row.loc[i, 'update'] = -1
                data_part1 = data.loc[0:row_index]
                data_part2 = data.loc[row_index + 1:]
                data = pd.concat([data_part1, deletion_row, data_part2], ignore_index=True)
    return data

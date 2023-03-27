import numpy as np
import pandas as pd
import time
from tqdm._tqdm import trange
from tqdm import tqdm


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


def process_data(data, frac=1, sparse_ratio=100, sparse=False):
    if sparse:
        data = sparse_data_new(data, sparse_ratio)
        data = insert_deletion_data(data, False)
    else:
        data = insert_deletion_data_original(data, False)
    return data

def sparse_data(data, frac=1, sparse_ratio=100):
    start = time.perf_counter()
    print('-------- Sparse data process starts --------')
    empty_data = pd.DataFrame(columns=data.columns, index=[i for i in range(sparse_ratio * len(data))])
    shuffle_data = pd.concat([data, empty_data]).sample(frac=frac).reset_index(drop=True)
    end = time.perf_counter()
    print('Sparse time cost: %d' % (start - end))
    print('-------- Sparse data process ends ---------')
    return shuffle_data


def sparse_data_new(data, sparse_ratio=100, shuffle=True):
    start = time.perf_counter()
    empty_data = pd.DataFrame(columns=data.columns, index=[i for i in trange(sparse_ratio * len(data))])
    sparse_data = empty_data
    for index in trange(data):
        row_index = np.random.randint(index, len(sparse_data))
    end = time.perf_counter()
    print('Sparse time costs: %ds' % (end - start))


def insert_deletion_data_original(data, concentrate=True):
    data = data.sample(frac=1).reset_index(drop=True)
    start = time.perf_counter()
    print('-------- Insert deletion data process starts ---------')
    if concentrate:
        data.insert(data.shape[1], 'update', np.nan)
        for i in trange(len(data)):
            if data.loc[i].all():
                continue
            else:
                data.loc[i, 'update'] = 1
                data.loc[len(data)] = data.loc[i]
                data.loc[len(data) - 1, 'update'] = -1
    else:
        data.insert(data.shape[1], 'update', np.nan)
        for i in tqdm(reversed(range(len(data)))):
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
    print('-------- Insert deletion data process ends ---------')
    end = time.perf_counter()
    print('Insert deletion data costs: %ds' % (end - start))
    return data


def insert_deletion_data(data, concentrate=True):
    start = time.perf_counter()
    print('-------- Insert deletion data process starts ---------')
    if concentrate:
        data.insert(data.shape[1], 'update', np.nan)
        for i in trange(len(data)):
            if data.loc[i].all():
                continue
            else:
                data.loc[i, 'update'] = 1
                data.loc[len(data)] = data.loc[i]
                data.loc[len(data) - 1, 'update'] = -1
    else:
        length = len(data)
        data.insert(data.shape[1], 'update', np.nan)
        for i in tqdm(reversed(range(len(data)))):
            # print('index %d' % i)
            if data.loc[i].all():
                continue
            else:
                data.loc[i, 'update'] = 1
                row_index = np.random.randint(i, len(data))
                deletion_row = data.loc[i:i].copy()
                deletion_row.loc[i, 'update'] = -1
                # print('row_index %d' % row_index)
                while row_index < length:
                    if pd.isnull(data.loc[row_index, 'update']):
                        break
                    else:
                        row_index += 1
                data.loc[row_index] = deletion_row.iloc[0]
        data.reset_index(drop=True)
    print('data: \n%s' % data)
    print('-------- Insert deletion data process ends ---------')
    end = time.perf_counter()
    print('Insert deletion data costs: %ds' % (end - start))
    return data

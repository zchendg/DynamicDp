import numpy as np
import pandas as pd
import time
from tqdm._tqdm import trange
from tqdm import tqdm

class DataLoader():
    def __init__(self, process_type, data, dynamic_size, delete_tail=True, sparsification=False, logger=None):
        assert process_type in ['original dataset', 'fixed size', 'insertion deletion', 'random delete']
        self.start = time.perf_counter()
        self.original_data = data
        if process_type == 'original dataset':
            self.dynamic_data = self.generate_original_data()
        elif process_type == 'fixed size':
            self.dynamic_data = self.generate_fixed_size_data(dynamic_size, delete_tail)
        elif process_type == 'insertion deletion':
            self.dynamic_data = self.generate_insertion_deletion_data()
        elif process_type == 'random delete':
            self.dynamic_data = self.generate_random_delete_data()
        self.insertion_only_data = self.generate_insertion_only_data()
        self.deletion_only_data = self.generate_deletion_only_data()
        self.end = time.perf_counter()
        logger.info('Process Type: %s, cost %ds' % (process_type, self.end - self.start))

    def generate_original_data(self):
        data = self.original_data
        data.insert(data.shape[1], 'update', 1)
        return data

    def generate_fixed_size_data(self, dynamic_size=1024, delete_tail=True):
        data = self.original_data
        data.insert(data.shape[1], 'update', 1)
        data_output = data[0: dynamic_size]
        data_current = data[0: dynamic_size]
        data = pd.concat([data, data[0: dynamic_size]], ignore_index=True).drop_duplicates(keep=False).reset_index(drop=True)
        for i in trange(len(data)):
            index = np.random.randint(0, dynamic_size)
            deletion_row = data_current.loc[index: index].copy()
            deletion_row.loc[:, 'update'] = -1
            insertion_row = data.loc[i: i].copy()
            data_output = pd.concat([data_output, deletion_row, insertion_row], ignore_index=True)
            deletion_row.loc[:, 'update'] = 1
            data_current = pd.concat([data_current, deletion_row, insertion_row], ignore_index=True).drop_duplicates(
                keep=False).reset_index(drop=True)
        if delete_tail:
            data_current['update'] = -1
            data_delete = data_current.sample(frac=1).reset_index(drop=True)
            data_output = pd.concat([data_output, data_delete], ignore_index=True).reset_index(drop=True)
        return data_output

    def generate_insertion_deletion_data(self):
        data = self.original_data
        data_insert = data
        data_delete = data
        data_insert(data.shape[1], 'update', 1)
        data_delete(data.shape[1], 'update', -1)
        data_insert = data_insert.sample(frac=1).reset_index(drop=True)
        data_delete = data_delete.sample(frac=1).reset_index(drop=True)
        data_output = pd.concat([data_insert, data_delete], ignore_index=True).reset_index(drop=True)
        return data_output

    def generate_random_delete_data(self):
        data = self.original_data
        length = len(data)
        data.insert(data.shape[1], 'update', np.nan)
        for i in tqdm(reversed(range(len(data)))):
            if data.loc[i].all():
                continue
            else:
                data.loc[i, 'update'] = 1
                row_index = np.random.randint(i, len(data))
                deletion_row = data.loc[i:i].copy()
                deletion_row.loc[i, 'update'] = -1
                while row_index < length:
                    if pd.isnull(data.loc[row_index, 'update']):
                        break
                    else:
                        row_index += 1
                data.loc[row_index] = deletion_row.iloc[0]
        data_output = data.reset_index(drop=True)
        return data_output

    def generate_insertion_only_data(self):
        insertion_only_data = self.dynamic_data
        for i in range(len(insertion_only_data)):
            if insertion_only_data.loc[i, 'update'] == -1:
                insertion_only_data.loc[i, 'update'] = np.nan
        return insertion_only_data

    def generate_deletion_only_data(self):
        deletion_only_data = self.dynamic_data
        for i in range(len(deletion_only_data)):
            if deletion_only_data.loc[i, 'update'] == 1:
                deletion_only_data.loc[i, 'update'] = np.nan
        return deletion_only_data

import numpy as np
import pandas as pd


class BasicCounting:
    def __init__(self, epsilon=1, delta=0):
        self.queries = []
        self.epsilon = epsilon
        self.delta = delta

    def generate_query_list(self):
        return self.queries

    def update_counting(self, df, DP=False):
        ctr = 0
        for row in df.itertuples():
            if row.update is not np.nan:
                ctr += 1
        ctr -= 1
        # When we want DP for the query, DP=True
        if DP:
            ctr += np.random.laplace(loc=0, scale=1 / self.epsilon)
        return ctr

    def create_query(self):
        return lambda df: self.update_counting(df)

    def create_DP_counting_query(self):
        return lambda df: self.update_counting(df, True)

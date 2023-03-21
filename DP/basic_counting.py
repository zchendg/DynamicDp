import numpy as np
import pandas as pd


class BasicCounting:
    def __init__(self, epsilon=1, delta=0, beta=0.05, store_df=False, config=None):
        self.queries = []
        self.epsilon = epsilon
        self.delta = delta
        self.beta = beta
        self.counter = 0
        if store_df:
            self.df = pd.DataFrame(columns=config.keys())

    def generate_query_list(self):
        return self.queries

    def update_counting(self, df, DP=False):
        for row in df.itertuples():
            if row.update is not np.nan:
                self.counter += 1
        self.counter -= 1
        # When we want DP for the query, DP=True
        if DP:
            self.counter += np.random.laplace(loc=0, scale=1 / self.epsilon)
        return self.counter

    def create_query(self):
        return lambda df: self.update_counting(df)

    def create_DP_counting_query(self):
        return lambda df: self.update_counting(df, True)


    def update(self, df):
        self.df = pd.concat([self.df, df]).drop_duplicates(keep=False)
        self.counter = len(self.df)

    # For dynamic tree
    def tilde_counter(self):
        return self.counter + np.random.laplace(loc=0, scale=1 / self.epsilon)

    def error_bound(self):
        error_bound = (1 / self.epsilon) * (np.log2(len(self.df) + 2) ** 1.5) * np.log2(1 / self.beta)
        return error_bound

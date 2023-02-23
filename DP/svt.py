import numpy as np
import pandas as pd


class SparseVectorMachine:
    def __init__(self, T):
        self.T = T

    # preserves epsilon-differential privacy
    def above_threshold(self, queries, df, T, epsilon):
        T_hat = T + np.random.laplace(loc=0, scale=2 / epsilon)

        for idx, q in enumerate(queries):
            nu_i = np.random.laplace(loc=0, scale=4 / epsilon)
            if q(df) + nu_i >= T_hat:
                return idx
        return None  # an invalid index

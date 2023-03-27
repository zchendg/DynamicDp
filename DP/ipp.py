import numpy as np
import pandas as pd
from svt import SparseVectorMachine
from basic_counting import BasicCounting

"""
In this file, I'd like to implement a method where whenever I add
one element into the current stream, I could update the partition
"""


class IPP:
    def __init__(self, df, epsilon, beta):
        self.s_0 = 0
        self.j = 1
        self.T = 2
        self.beta = 6 * beta / np.square(np.pi)
        self.C = 7 * np.log(2 * self.T / self.beta) / epsilon
        self.epsilon = epsilon
        self.SVT = SparseVectorMachine(self.C)
        self.df = df
        self.segment = [0]

    def update_segment(self, t):
        # here the query is just counting query for the number
        queries = [BasicCounting(self.epsilon).create_query()]
        # query asks |{sj-1 + 1 <= i <= t: xi not void}|
        cur_df = self.df[self.segment[self.j - 1] + 1: t + 1]
        output = self.SVT.above_threshold(queries, cur_df, self.C, self.epsilon)
        if not output is None or t >= self.T:
            self.segment.append(t)
            self.j += 1
            self.T = t * t
            self.beta = 6 * self.beta / (np.square(np.pi) * np.square(self.j))
            temp = 2 * self.T / self.beta
            self.C = 7 * np.log(temp) / self.epsilon
            self.SVT = SparseVectorMachine(self.C)

    def get_segment(self):
        return self.segment

    def __repr__(self):
        return 's0: %s, j: %s, T: %s, beta: %s, C: %s, epsilon: %s, \n segment: %s' % (
            self.s_0, self.j, self.T, self.beta, self.C, self.epsilon, self.segment)

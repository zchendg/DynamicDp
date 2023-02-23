import numpy as np
import pandas as pd
import math
from basic_counting import BasicCounting
from mbi import Dataset, FactoredInference, Domain


class Node:
    # node_index is vi
    def __init__(self, node_index, df, segments, beta, epsilon, delta, node_tree):
        self.node_index = node_index
        self.segments = segments
        self.beta = beta
        self.epsilon = epsilon
        self.delta = delta
        self.height = 0
        temp = self.node_index
        for height in range(0, int(math.log(self.node_index, 2))):
            if(temp % 2 == 0): self.height += 1
        self.items = self.store_items(self, df, node_tree)

    # Use for initialization, when create the node, we need to store the item in to the nodes
    # There are still some optimization methods, We shall do it later.
    def store_items(self, df, node_tree):
        # Seperate the node into two cases: left most and non left most
        items_stored = []
        if(self.get_left_ancestor(self) is None):
            for line in df:
                if(line[0] == 0): continue
                elif(line[0] == 1): items_stored += line
                elif(line[0] == -1 and line in items_stored): items_stored.remove[line]
            return items_stored
        else: 
            ancestor_item = node_tree[self.get_left_ancestor(self)].items
            # The line is stores only if the row is not stored in the node's ancestor
            for line in df:
                if(line in ancestor_item): continue
                elif(line[0] == 0): continue
                elif(line[0] == 1): items_stored += line
                elif(line[0] == -1 and line in items_stored): items_stored.remove[line]
            return items_stored

    def insertion_mechanism(df, beta, epsilon, delta):
        return

    def counting_query(df, beta, epsilon, delta):
        return BasicCounting.update_counting(df)

    # Get left closest ancestor, if ther is no left closest ancestor, return None
    # For node 1, 2, 4, 8... there is no left ancestor exist, so return None
    # For the remaining nodes, the ancestor index is node_index - 2^height
    def get_left_ancestor(self):
        quotient = self.node_index
        while True:
            if(quotient == 1): return None
            elif(quotient < 1): break
            else: quotient = self.node_index / 2
        return self.node_index - 2 ** self.height

    # When making the query, we give the timestamp and need to determine
    # which segment that the timestamp belongs to
    def get_query_index(self, timestamp):
        for idx in range(len(self.segments)):
            if(timestamp > self.segments[idx] and timestamp <= self.segments[idx]):
                return idx
        return None

    def node_query(self, timestamp):
        # Derive while segment that timestamp belongs to
        j = self.get_index(self, timestamp)
        while True:
            if(j == self.node_index):
                # Here need to write D(v) All items in Dsj inserted after the closet left-ancestor of v
                r = 1
                epsilon= 3 * self.epsilon / (2 * np.square(np.pi) * r *r)
                delta= 2 * self.delta / (np.square(np.pi) * r *r)
                # The node query is simply the counting query
                n_wave = BasicCounting(self.df) + np.random.laplace(loc=0, scale = 1/epsilon)
                # Run (epsilon_r, delta_r)-DP to release Q(D(v))
            elif(j > self.node_index):
                for update in self.df:
                    # Here update (x, y) in segment sj
                    if(update[1] == -1 and update )
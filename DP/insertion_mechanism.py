import sys
import pandas as pd
import json
import logging
import os
import argparse
import auxiliary
from node import Node
from mbi import Domain, Dataset

# This mechanmism implement the baseline mechanism, that using insertion and deletion only mechanmism

class Insertion_Mechanism:
    def __init__(self, config):
        self.node_list = [Node(0, config.keys(), 0)]
        self.config = config
        self.domain = Domain(config.keys(), config.values())
        self.query_instance

    def create_node(self, sj):
        self.node_list.append(Node(len(self.node_list), self.config.keys(), sj))

    def df_after_closest_left_ancestor(self, index, current_df):
        set_diff_df = current_df.data
        for node in self.query_nodes(index):
            set_diff_df = pd.concat([set_diff_df, node.df, node.df]).drop_duplicates(keep=False)
        return set_diff_df

    def create_leftmost_node(self, cur_data, sj):
        self.create_node(sj)
        self.node_list[-1].add_items(cur_data.data)

    def create_internal_node(self, cur_data, sj):
        self.create_node(sj)
        diff_df = self.df_after_closest_left_ancestor(len(self.node_list) - 1, cur_data)
        self.node_list[-1].add_items(diff_df)

    def query_nodes(self, index):
        nodes = [self.node_list[index]]
        if auxiliary.is_two_power(index):
            return nodes
        else:
            return nodes + self.query_nodes(auxiliary.find_far_left_ancestor_index(self.node_list[index].index, self.node_list[index].height))

    def testing(self, ipp_instance, column_number=1, each_query_size=10, epsilon=1, delta=0, beta=0.05, iteration=500, logger=None):
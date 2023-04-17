import sys
import pandas as pd
import numpy as np
import json
import logging
import os
import argparse
import auxiliary
from node import Node
from approximation_instance import ApproximationInstance
from mbi import Domain, Dataset


# This mechanmism implement the baseline mechanism, that using insertion and deletion only mechanmism

class Insertion_Mechanism:
    def __init__(self, config, query_instance=None):
        self.node_list = [Node(0, config.keys(), 0)]
        self.config = config
        self.domain = Domain(config.keys(), config.values())
        self.query_instance = query_instance
        self.answer_ground_truth = {}
        self.answer_mechanism = {}
        self.initialize_answer()

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
            return nodes + self.query_nodes(
                auxiliary.find_far_left_ancestor_index(self.node_list[index].index, self.node_list[index].height))

    def testing(self, ipp_instance, column_number=1, each_query_size=10, epsilon=1, delta=0, beta=0.05, iteration=500,
                logger=None):
        for index in range(1, len(ipp_instance.get_segment()) - 1):
            logger.info('++++++++ Testing on node %d Started ++++++++' % index)

    def testing_index(self, index, query_instance, epsilon=1, delta=0, beta=0.05, iteration=500, logger=None):
        for member in self.query_instance.queries.keys():
            self.answer_ground_truth[member][index] = self.answer_queries_ground_truth(index,
                                                                                       self.query_instance.queries,
                                                                                       member, logger)
            self.answer_mechanism[member][index] = self.answer_queries_baseline()
        return

    def answer_queries_ground_truth(self, cur_index, queries, member, logger=None):
        current_dataset = self.find_current_dataset(cur_index)
        answer_ground_truth = self.answer_queries(current_dataset, member, queries)
        return np.array(answer_ground_truth)

    def answer_queries_baseline(self, cur_index, queries, member, epsilon=1, delta=0, iteration=500, logger=None):
        current_dataset = self.find_current_dataset(cur_index)
        approximate_instance = ApproximationInstance(current_dataset, self.domain, epsilon, [member], 'Data', iteration)
        answer_baseline = self.answer_queries(approximate_instance.approximated_data.df, member, queries)
        return np.array(answer_baseline)

    def answer_queries_dict(self, dataset, member, queries):
        answer = {}
        for length in queries[member].keys():
            answer[length] = []
            for query in queries[member][length]:
                answer[length] += [len(query(dataset))]
        return answer

    def answer_queries(self, dataset, member, queries):
        answer = []
        for length in queries[member].keys():
            for query in queries[member][length]:
                answer += [len(query(dataset))]
        return answer

    def find_current_dataset(self, cur_index, logger=None):
        Dv_list = []
        for node in self.node_list[1:]:
            if node.index <= cur_index:
                Dv_list += [node.df]
        current_dataset = pd.concat(Dv_list).drop_duplicates(keep=False)
        return current_dataset

    def initialize_answer(self):
        for member in self.query_instance.queries.keys():
            self.answer_ground_truth[member] = {}
            self.answer_mechanism[member] = {}
        return

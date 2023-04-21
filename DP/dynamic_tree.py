import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from node import Node
from query import Query
from approximation_instance import ApproximationInstance
from mbi import Domain, Dataset
from basic_counting import BasicCounting
import auxiliary
import auxiliary1
from datetime import datetime


class DynamicTree:
    def __init__(self, config, query_instance=None):
        self.node_list = [Node(0, config.keys(), 0)]
        self.config = config
        self.domain = Domain(config.keys(), config.values())
        self.answer_ground_truth = {}
        self.answer_golden_standard = {}
        self.answer_mechanism = {}
        self.query_instance = query_instance

    def insert_item(self, index, item):
        self.node_list[index - 1].add_item(item)

    def create_node(self, sj):
        # Create a node at the end of the tree in the sequential time order
        self.node_list.append(Node(len(self.node_list), self.config.keys(), sj))

    def df_after_closest_left_ancestor(self, index, current_df):
        set_diff_df = current_df.data
        for node in self.query_nodes(index):
            set_diff_df = pd.concat([set_diff_df, node.df, node.df]).drop_duplicates(keep=False)
        return set_diff_df

    def create_leftmost_node(self, cur_data, delete_data, sj):
        self.create_node(sj)
        self.node_list[-1].add_items(cur_data.data)
        self.node_list[-1].add_items(delete_data.data, delete_df=True)

    def create_internal_node(self, cur_data, delete_data, sj):
        self.create_node(sj)
        diff_df = self.df_after_closest_left_ancestor(len(self.node_list) - 1, cur_data)
        self.node_list[-1].add_items(diff_df)
        self.node_list[-1].add_items(delete_data.data, delete_df=True)

    def find_far_left_ancestor_index(self, index, height):
        if index / (2 ** height) % 4 == 1:
            return self.find_far_left_ancestor_index(index + 2 ** height, height + 1)
        elif index / (2 ** height) % 4 == 3:
            return index - 2 ** height

    # Below is the function serves to queries
    def query_nodes(self, index):
        nodes = [self.node_list[index]]
        if auxiliary.is_two_power(index):
            return nodes
        else:
            return nodes + self.query_nodes(
                self.find_far_left_ancestor_index(self.node_list[index].index, self.node_list[index].height))

    # Implement testing function in the data structure
    def testing(self, ipp_instance, column_number=1, each_query_size=10, epsilon=1, delta=0, beta=0.05, iteration=500,
                logger=None):
        for member in self.query_instance.queries.keys():
            self.answer_ground_truth[member] = {}
            self.answer_golden_standard[member] = {}
            self.answer_mechanism[member] = {}
        for index in range(1, len(ipp_instance.get_segment()) - 1):
            logger.info('++++++++ Testing on node %d Started ++++++++' % index)
            self.testing_index(index, epsilon, delta, beta, iteration, logger)
            logger.info('++++++++ Testing on node %d Finished ++++++++' % index)

    # Implement testing for at particular position, this function
    def testing_index(self, index, epsilon=1, delta=0, beta=0.05, iteration=500,
                      logger=None):
        query_nodes = self.query_nodes(index)
        query_nodes.reverse()
        logger.info(
            'At node with index %d, we implement queries on cliques %s:' % (index, self.query_instance.queries.keys()))
        for member in self.query_instance.queries.keys():
            self.answer_ground_truth[member][index] = self.answer_queries_ground_truth(query_nodes, index,
                                                                                       self.query_instance.queries,
                                                                                       member,
                                                                                       logger=logger)
            self.answer_golden_standard[member][index] = self.answer_queries_golden_standard(query_nodes, index,
                                                                                             self.query_instance.queries,
                                                                                             member,
                                                                                             epsilon, delta, iteration,
                                                                                             logger=logger)
            self.answer_mechanism[member][index] = self.answer_queries_mechanism(query_nodes, index,
                                                                                 self.query_instance.queries, member,
                                                                                 epsilon, delta, beta, iteration,
                                                                                 logger=logger)
            logger.info('The testing is implemented at %s' % member)
            logger.info('Ground truth: gives answer')
            auxiliary1.output_answer(self.answer_ground_truth[member][index], member, self.query_instance, logger)
            logger.info('Golden standard: gives answer')
            auxiliary1.output_answer(self.answer_golden_standard[member][index], member, self.query_instance, logger)
            logger.info('Mechanism: gives answer')
            auxiliary1.output_answer(self.answer_mechanism[member][index], member, self.query_instance, logger)
            logger.info("Mean Square Error of ground truth and golden standard: %s" % self.mse(
                self.answer_ground_truth[member][index],
                self.answer_golden_standard[member][index]))
            logger.info(
                "Mean Square Error of ground truth and mechanism: %s" % self.mse(
                    self.answer_ground_truth[member][index], self.answer_mechanism[member][index]))
            logger.info("Mean Square Error of golden standard and mechanism: %s" % self.mse(
                self.answer_golden_standard[member][index],
                self.answer_mechanism[member][index]))
            # logger.info('Measurement1: ground_truth and golden_standard\n' + str( self.compare_results(
            # self.answer_ground_truth[member], self.answer_golden_standard[member], measurement=3))) logger.info(
            # 'Measurement2: ground_truth and mechanism\n' + str( self.compare_results(self.answer_ground_truth[
            # member], self.answer_mechanism[member], measurement=3)))
            logger.info('Measurement3: golden_standard and mechanism\n' + str(
                auxiliary1.compare_results(self.answer_golden_standard[member], self.answer_mechanism[member],
                                           measurement=3)))

    # For ground truth, returns the queries' answer for original data.
    def answer_queries_ground_truth(self, nodes, cur_index, queries, member, logger=None):
        Dv_list = []
        for node in nodes:
            Dv = pd.DataFrame(columns=self.config.keys())
            delete_df = pd.DataFrame(columns=self.config.keys())
            for index in range(cur_index + 1):
                if index == node.index:
                    Dv = node.df
                elif node.index < index <= cur_index:
                    delete_df = pd.merge(delete_df, self.node_list[index].delete_df, how='outer')
            Dv = pd.concat([Dv, delete_df, delete_df]).drop_duplicates(keep=False)
            Dv_list += [Dv]
        Dataset_r = pd.concat(Dv_list).drop_duplicates(keep='first')
        answer_ground_truth = self.answer_queries(Dataset_r, member, queries)
        return np.array(answer_ground_truth)

    # For golden standard, returns the queries' answer for golden standard (Approximated data)
    def answer_queries_golden_standard(self, nodes, cur_index, queries, member, epsilon=1, delta=0, iteration=500,
                                       logger=None):
        Dv_list = []
        for node in nodes:
            Dv = pd.DataFrame(columns=self.config.keys())
            delete_df = pd.DataFrame(columns=self.config.keys())
            for index in range(cur_index + 1):
                if index == node.index:
                    Dv = node.df
                elif node.index < index <= cur_index:
                    delete_df = pd.merge(delete_df, self.node_list[index].delete_df, how='outer')
            Dv = pd.concat([Dv, delete_df, delete_df]).drop_duplicates(keep=False)
            Dv_list += [Dv]
        Dataset_r = pd.concat(Dv_list).drop_duplicates(keep='first')
        approximate_instance = ApproximationInstance(Dataset_r, self.domain, epsilon, [member], 'Data', iteration)
        answer_golden_standard = self.answer_queries(approximate_instance.approximated_data.df, member, queries)
        return np.array(answer_golden_standard)

    # For new algorithm, returns the queries' answer for the new algorithm
    def answer_queries_mechanism(self, nodes, cur_index, queries, member, epsilon=1, delta=0, beta=0.05, iteration=500,
                                 logger=None):
        epsilon_budget = {node: 6 * epsilon / (np.square(np.pi * (node.height + 1))) for node in nodes}
        delta_budget = {node: 6 * delta / (np.square(np.pi * (node.height + 1))) for node in nodes}
        answer_mechanism = self.answer_queries(pd.DataFrame(columns=self.config.keys()), member, queries)
        for node in nodes:
            answer_mechanism_node = self.answer_node_queries_mechanism(node, cur_index, queries, member,
                                                                       epsilon_budget[node], delta_budget[node],
                                                                       beta, iteration, logger)
            answer_mechanism = np.array(answer_mechanism) + np.array(answer_mechanism_node)
        return np.array(answer_mechanism)

    def answer_node_queries_mechanism(self, node, cur_index, queries, member, epsilon=1, delta=0, beta=0.05,
                                      iteration=500, logger=None):
        logger.info('-------- New Mechanism: Testing on %s, node %s is accessed --------' % (cur_index, node.index))
        # Initiate delete_df to store the data in the deletion-only problem
        tilde_n_v = 0
        Dv = pd.DataFrame(columns=self.config.keys())
        delete_df = pd.DataFrame(columns=self.config.keys())
        # These variables store the query answer for the approximated dataset
        Dv_answer = []
        answer_mechanism = []
        for index in range(cur_index + 1):
            if index == node.index:
                Dv = node.df
                r = 1
                if len(Dv) <= 1:
                    epsilon_r = epsilon
                    delta_r = delta
                    Dv_answer = self.answer_queries(Dv, member, queries)
                    answer_mechanism = Dv_answer
                    # Initiate M_BC
                    basic_counting_instance = BasicCounting(epsilon, delta_r, store_df=True, config=self.config)
                    continue
                # Epsilon_r might be modified, as the number of restarts is fixed
                epsilon_r = max(3 * epsilon / (2 * np.square(np.pi * r)), epsilon / np.log2(len(Dv)))
                delta_r = max(2 * delta / (np.square(np.pi * r)), delta / np.log2(len(Dv)))
                tilde_n_v = len(Dv) + np.random.laplace(loc=0, scale=1 / epsilon_r)
                approximation_instance = ApproximationInstance(Dv, self.domain, epsilon_r, [member], 'Data', iteration)
                Dv_answer = self.answer_queries(approximation_instance.approximated_data.df, member, queries)
                answer_mechanism = Dv_answer
                # Initiate M_Ins
                basic_counting_instance = BasicCounting(epsilon, delta_r, store_df=True, config=self.config)
            elif node.index < index <= cur_index:
                # delete_df contains the actual item that has been deleted.
                delete_df = pd.merge(pd.concat([delete_df, self.node_list[index].delete_df]).drop_duplicates(keep='first'), Dv, how='inner')
                basic_counting_instance.update(pd.merge(self.node_list[index].delete_df, Dv, how='inner'))
                tilde_n_del = basic_counting_instance.tilde_counter()
                # The error bound of MBC at time sj
                alpha_BC_sj = basic_counting_instance.error_bound()
                if tilde_n_del > (tilde_n_v / 2 + 2 * alpha_BC_sj):
                    # Remove all augmented items from D(v)
                    # Dv = pd.concat([Dv, delete_df, delete_df]).drop_duplicates(keep=False)
                    r = r + 1
                    if len(Dv) <= 1:
                        # epsilon_r and delta_r remains unchanged
                        tilde_n_v = len(Dv) + np.random.laplace(loc=0, scale=1 / epsilon_r)
                        approximation_instance = ApproximationInstance(Dv, self.domain, epsilon_r, [member], 'Data',
                                                                       iteration)
                        Dv_answer = self.answer_queries(approximation_instance.approximated_data.df, member, queries)
                        if tilde_n_v < (2 * alpha_BC_sj):
                            ansewr_mechanism = self.ansewr_queries(pd.DataFrame(columns=self.config.keys()), member,
                                                                   queries)
                            break
                    epsilon_r = max(3 * epsilon / (2 * np.square(np.pi * r)), epsilon / np.log2(len(Dv)))
                    delta_r = max(2 * delta / (np.square(np.pi * r)), delta / np.log2(len(Dv)))
                    tilde_n_v = len(Dv) + np.random.laplace(loc=0, scale=1 / epsilon_r)
                    # Run(epsilon_r, delta_r)-DP M(D(v)) to release Q(D(v))
                    approximation_instance = ApproximationInstance(Dv, self.domain, epsilon_r, [member], 'Data',
                                                                   iteration)
                    # Re-publish the query answer for Dv
                    Dv_answer = self.answer_queries(approximation_instance.approximated_data.df, member, queries)
                    # When this condition happens, we all the Q(D_t(v)) will return 0, hence just
                    if tilde_n_v < (2 * alpha_BC_sj):
                        logger.info('Testing on index %d, mechanism passed index %d, the algorithm hit tilde_n_del > ('
                                    'tilde_n_v / 2 + 2 * alpha_BC_sj, with %f > (%f / 2 + 2 * %f)' % (
                                        node.index, index, tilde_n_del, tilde_n_v, alpha_BC_sj))
                        answer_mechanism = np.array(Dv_answer)
                        break
                else:
                    D_sj_answer = self.answer_queries(delete_df, member, queries)
                    answer_mechanism = np.array(Dv_answer) - np.array(D_sj_answer)
                    if index == cur_index:
                        break
        logger.info('-------- New Mechanism answer node query on node %s finished --------' % node.index)
        answer_mechanism = [max(answer, 0) for answer in answer_mechanism]
        return np.array(answer_mechanism)

    # Pack the way for answering the query.
    # queries: being a dict that maps the member to a dict, for the contained dict, key is length, value is the queries
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

    # Divide queries into group with different length
    def output_answer(self, answer, member, query_instance, logger=None):
        head = 0
        tail = 0
        logger.info('output_answer:%s' % answer)
        for length in query_instance.length_size[member]:
            tail = query_instance.length_size[member][length]
            if logger is not None:
                logger.info('Range size %d, answer %s:' % (length, str(answer[head: tail])))
            else:
                print('Range size %d, answer:' % length)
                print(str(answer[head: tail]))
            head = query_instance.length_size[member][length]

    def compare_results(self, answer1, answer2, measurement=1):
        # For golden standard, returns the queries' answer with pgm mechanism on single static dataset.
        # Parameter: measurement:
        # measurement=1: give the average of the difference divides the golden_standard
        # measurement=2: give the mean difference between the queries
        m_answer1 = [i + 1 for i in answer1]
        m_answer2 = [i + 1 for i in answer2]
        if measurement == 1:
            difference = np.absolute(np.array(m_answer1) - np.array(m_answer2))
            result = np.array(difference) / np.array(m_answer2)
            return np.mean(result)
        if measurement == 2:
            return (np.mean(m_answer1)) / np.mean(m_answer2)
        if measurement == 3:
            difference = np.absolute(np.array(m_answer1) - np.array(m_answer2))
            return np.sum(difference) / np.sum(m_answer1)
        return -1

    def mse(self, array1, array2):
        return ((np.array(array1) - np.array(array2)) ** 2).mean()

    def update_answer_data(self, member, index, answer1, answer2, answer3):
        self.answer_queries_ground_truth[member][index] = answer1
        self.answer_golden_standard[member][index] = answer2
        self.answer_mechanism[member][index] = answer3

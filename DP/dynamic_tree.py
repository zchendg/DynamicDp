import numpy as np
import pandas as pd
from node import Node
from query import Query
from approximation_instance import ApproximationInstance
from mbi import Domain


class DynamicTree:
    def __init__(self, config):
        self.node_list = [Node(0, config.keys())]
        self.config = config
        self.domain = Domain(config.keys(), config.values())

    def insert_item(self, index, item):
        self.node_list[index - 1].add_item(item)

    def create_node(self):
        # Create a node at the end of the tree in the sequential time order
        self.node_list.append(Node(len(self.node_list), self.config.keys()))

    def df_after_closest_left_ancestor(self, index, current_df):
        set_diff_df = current_df.data
        for node in self.query_nodes(index):
            set_diff_df = pd.concat([set_diff_df, node.df, node.df]).drop_duplicates(keep=False)
        return set_diff_df

    def create_leftmost_node(self, cur_data, delete_data):
        self.create_node()
        self.node_list[-1].add_items(cur_data.data)
        self.node_list[-1].add_items(delete_data.data, delete_df=True)

    def create_internal_node(self, cur_data, delete_data):
        self.create_node()
        diff_df = self.df_after_closest_left_ancestor(len(self.node_list) - 1, cur_data)
        self.node_list[-1].add_items(diff_df)
        self.node_list[-1].add_items(delete_data.data, delete_df=True)

    def find_far_left_ancestor_index(self, index, height):
        if index / (2 ** height) % 4 == 1:
            return self.find_far_left_ancestor_index(index + 2 ** height, height + 1)
        elif index / (2 ** height) % 4 == 3:
            return index - 2 ** height

    def is_two_power(self, n):
        if n == 1:
            return True
        res = 1
        while res < n:
            res = res << 1
        if res == n:
            return True
        else:
            return False

    # Below is the function serves to queries
    def query_nodes(self, index):
        nodes = [self.node_list[index]]
        if self.is_two_power(index):
            return nodes
        else:
            return nodes + self.query_nodes(
                self.find_far_left_ancestor_index(self.node_list[index].index, self.node_list[index].height))

    # Implement testing function in the data structure
    def testing(self, ipp_instance, column_number=1, each_query_size=100, epsilon=1, delta=0, iteration=500,
                logger=None):
        query_instance = Query(self.config, column_number, each_query_size)
        for index in range(1, len(ipp_instance.get_segment()) - 1):
            query_nodes = self.query_nodes(index)
            query_nodes.reverse()
            logger.info(
                'At node with index %d, we implement queries on cliques %s:' % (index, query_instance.queries.keys()))
            logger.info('Each clique consists of %d queries' % each_query_size)
            for member in query_instance.queries.keys():
                answer = self.answer_queries(query_nodes, index, query_instance.queries[member], member, epsilon, delta,
                                             iteration)
                answer_golden_standard = self.answer_queries_golden_standard(query_nodes, index,
                                                                             query_instance.queries[member], member,
                                                                             epsilon, delta, iteration)
                mse = ((np.array(answer) - np.array(answer_golden_standard)) ** 2).mean()
                logger.info('The testing is implemented at %s' % member)
                logger.info('Our mechanism gives answer: \n' + str(answer))
                logger.info('Static mechanism gives answer (golden standard): \n' + str(answer_golden_standard))
                logger.info("Mean Square Error: %s" % mse)
                logger.info('Measurement1: ' + str(self.compare_results(answer, answer_golden_standard, measurement=1)))
                logger.info('Measurement2: ' + str(self.compare_results(answer, answer_golden_standard, measurement=2)))

    def answer_queries(self, nodes, cur_index, queries, member, epsilon=1, delta=0, iteration=500):
        epsilon_budge = {node: 6 * epsilon / (np.square(np.pi * (node.height + 1))) for node in nodes}
        delta_budge = {node: 6 * delta / (np.square(np.pi * (node.height + 1))) for node in nodes}
        answer = [0] * len(queries)
        for node in nodes:
            queries_answer = self.answer_node_queries(node, cur_index, queries, member,
                                                      epsilon_budge[node],
                                                      delta_budge[node], iteration)
            answer = np.array(answer) + np.array(queries_answer)
        return answer

    def answer_node_queries(self, node, cur_index, queries, member, epsilon=1, delta=0, beta=0.05, iteration=500):
        # Initiate new object delete_df to store the data in the deletion-only problem
        global widetilde_n_v
        global D_v
        delete_df = pd.DataFrame(columns=self.config.keys())
        # These variables store the query answer for the approximated dataset
        D_v_answer = []
        D_sj_answer = []
        answer = []
        for index in range(cur_index + 1):
            if index == node.index:
                D_v = node.df
                r = 1
                epsilon_r, delta_r = 3 * epsilon / (2 * np.square(np.pi * r)), 2 * delta / (
                        2 * np.square(np.pi * r))
                widetilde_n_v = len(D_v) + np.random.laplace(loc=0, scale=1 / epsilon_r)
                approximation_instance = ApproximationInstance(D_v, self.domain, epsilon_r, [member], 'Data', iteration)
                for query in queries:
                    D_v_answer += [len(query(approximation_instance.approximated_data.df))]
                    answer = D_v_answer
            elif node.index < index <= cur_index:
                # Currently, I omit the restart procedure for simplicity
                delete_df = pd.merge(
                    pd.concat([delete_df, self.node_list[index].delete_df]).drop_duplicates(keep=False), D_v)
                widetilde_n_del = len(delete_df)
                # The error bound of MBC at time sj
                alpha_BC_sj = (1 / epsilon) * (np.log2(self.node_list[index].sj) ** 1.5) * np.log2(1 / beta)
                if widetilde_n_del > (widetilde_n_v / 2 + 2 * alpha_BC_sj):
                    # Remove all augmented items from D(v)
                    D_v = pd.concat([D_v, delete_df, delete_df]).drop_duplicates(keep=False)
                    r = r + 1
                    epsilon_r, delta_r = 3 * epsilon / (2 * np.square(np.pi * r)), 2 * delta / (
                            2 * np.square(np.pi * r))
                    widetilde_n_v = len(D_v) + np.random.laplace(loc=0, scale=1 / epsilon_r)
                    # Run(epsilon_r, delta_r)-DP M(D(v)) to release Q(D(v))
                    approximation_instance = ApproximationInstance(D_v, self.domain, epsilon_r, [member], 'Data',
                                                                   iteration)
                    # Re-publish the query answer for D_v
                    D_v_answer = []
                    for query in queries:
                        D_v_answer += [len(query(approximation_instance.approximated_data.df))]
                    # When this condition happens, we all the Q(D_t(v)) will return 0, hence just
                    if widetilde_n_v < (2 * alpha_BC_sj):
                        answer = np.array(D_v_answer)
                        return answer
                elif index == cur_index:
                    approximation_instance_delete = ApproximationInstance(delete_df, self.domain, epsilon_r, [member],
                                                                          'Data', iteration)
                    for query in queries:
                        D_sj_answer += [len(query(approximation_instance_delete.approximated_data.df))]
                    answer = np.array(D_v_answer) - np.array(D_sj_answer)
                    return answer
        return answer

    # But if the result is not good, we can compute the golden standard by summing the result in different
    # node up
    # Tomorrow you should complete the full version with re-start

    def answer_queries_golden_standard(self, nodes, cur_index, queries, member, epsilon=1, delta=0, iteration=500):
        Dv_list = []
        for node in nodes:
            Dv = pd.DataFrame(columns=self.config.keys())
            for index in range(cur_index + 1):
                if index == node.index:
                    Dv = node.df
                elif node.index < index <= cur_index:
                    Dv = pd.concat(
                        [D_v, self.node_list[index].delete_df, self.node_list[index].delete_df]).drop_duplicates(
                        keep=False)
            Dv_list += [Dv]
        Dataset = pd.concat(Dv_list).drop_duplicates(keep=False)
        approximate_instance_golden_standard = ApproximationInstance(Dataset, self.domain, epsilon, [member], 'Data',
                                                                     iteration)
        answer_golden_standard = []
        for query in queries:
            answer_golden_standard += [len(query(approximate_instance_golden_standard.approximated_data.df))]
        return np.array(answer_golden_standard)

    def compare_results(self, answer, answer_golden_standard, measurement=1):
        # Parameter: measurement:
        # measurement=1: give the average of the difference divides the golden_standard
        # measurement=2: give the mean difference between the queries
        m_answer = [i + 1 for i in answer]
        m_answer_golden_standard = [i + 1 for i in answer_golden_standard]
        if measurement == 1:
            difference = np.absolute(np.array(m_answer) - np.array(m_answer_golden_standard))
            result = np.array(difference) / np.array(m_answer_golden_standard)
            return np.mean(result)
        if measurement == 2:
            return (np.mean(m_answer)) / np.mean(m_answer_golden_standard)
        return -1

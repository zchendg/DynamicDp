import numpy as np
import pandas as pd
from node import Node
from query import Query
from approximation_instance import ApproximationInstance
from mbi import Domain


def find_far_left_ancestor_index(index, height):
    if index / (2 ** height) % 4 == 1:
        return find_far_left_ancestor_index(index + 2 ** height, height + 1)
    elif index / (2 ** height) % 4 == 3:
        return index - 2 ** height


def is_two_power(n):
    if n == 1:
        return True
    res = 1
    while res < n:
        res = res << 1
    if res == n:
        return True
    else:
        return False


class Dynamic_Tree:
    def __init__(self, config):
        self.node_list = [Node(0, config.keys())]
        self.config = config
        self.domain = Domain(config.keys(), config.values())

    def insert_item(self, index, item):
        self.node_list[index - 1].add_item(item)

    def create_node(self, index):
        self.node_list += Node(index)

    def df_after_closest_left_ancestor(self, index, current_df):
        set_diff_df = current_df.data
        for node in self.query_nodes(index):
            set_diff_df = pd.concat([set_diff_df, node.df, node.df]).drop_duplicates(keep=False)
        return set_diff_df

    def query_nodes(self, index):
        nodes = [self.node_list[index]]
        if is_two_power(len(self.node_list) - 1):
            return nodes
        else:
            return nodes + self.query_nodes(
                find_far_left_ancestor_index(self.node_list[index].index, self.node_list[index].height))

    # Implement testing function in the data structure
    def testing(self, ipp_instance, column_number=1, each_query_size=100, epsilon=0):
        query_instance = Query(self.config, column_number, each_query_size)
        for index in range(1, len(ipp_instance.get_segment()) - 1):
            query_nodes = self.query_nodes(index)
            query_nodes.reserve()
            # Logging function required here
            for member in query_instance.queries.keys():
                answer, answer_golden_standard = self.a

    def answer_queries(self, nodes, cur_index, queries, member, epsilon=1, delta=0, iteration=500):
        epsilon_budge = {node: 6 * epsilon / (np.square(np.pi * (node.height + 1))) for node in nodes}
        delta_budge = {node: 6 * delta / (np.square(np.pi * (node.height + 1))) for node in nodes}
        answer = [0] * len(queries)
        for node in nodes:
            queries_answer, queries_answer_golden_standard = self.answer_node_queries(node, cur_index, queries, member, epsilon_budge[node], delta_budge[node], iteration)
            answer = np.array(answer) + np.array(queries_answer)
            answer_golden_standard = np.array(answer_golden_standard) + np.array(queries_answer_golden_standard)
        return answer, answer_golden_standard

    def answer_node_queries(self, node, cur_index, queries, member, epsilon=1, delta=0, beta=0.05, iteration=500):
        # Initiate new object delete_df to store the data in the deletion-only problem
        global widetilde_n_v
        delete_df = pd.DataFrame(columns=self.config.keys())
        # These variables store the query answer for the approximated dataset
        D_v_answer = []
        D_sj_answer = []
        # These variables store the query answer for the ground truth
        D_v_answer_golden_standard = []
        D_sj_answer_golden_standard = []
        for index in range(cur_index + 1):
            if index == node.index:
                node_df = node.df
                r = 1
                epsilon_r, delta_r = 3 * epsilon[node] / (2 * np.square(np.pi * r)), 2 * delta[node] / (
                        2 * np.square(np.pi * r))
                widetilde_n_v = len(node_df) + np.random.laplace(loc=0, scale=1 / epsilon_r)
                approximation_instance = ApproximationInstance(node_df, self.domain, 1, [member], 'Data', iteration)
                for query in queries:
                    D_v_answer += [len(query(approximation_instance.approximated_data.df))]
                    # D_v_answer_golden_standard += [len(query(node_df))]
            elif node.index < index <= cur_index:
                # Currently, I omit the restart procedure for simplicity
                delete_df = pd.merge(self.node_list[index].delete_df, node_df)
                node_df = pd.concat([node_df, delete_df, delete_df]).drop_duplicates(keep=False)
                widetilde_n_del = len(delete_df)
                alpha_BC_sj = (1/epsilon) * (np.log2(self.node_list[index].sj) ** 1.5) * np.log2(1/beta)
                if widetilde_n_del > (widetilde_n_v / 2 + 2 * alpha_BC_sj):
                    r = r + 1
                    epsilon_r, delta_r = 3 * epsilon[node] / (2 * np.square(np.pi * r)), 2 * delta[node] / (
                            2 * np.square(np.pi * r))
                    widetilde_n_v = len(node_df) + np.random.laplace(loc=0, scale=1 / epsilon_r)
                #     Remains to write Run(epsilon_r, delta_r)-DP M(D(v)) to release Q(D(v))
                else:
                # print("node_list length: ", len(node_list), " index: ", index)
                delete_df = pd.concat([delete_df, self.node_list[index].delete_df]).drop_duplicates(keep=False)
        deleted_df = pd.concat([node_df, delete_df, delete_df]).drop_duplicates(keep=False)
        approximation_instance_delete = ApproximationInstance(deleted_df, self.domain, 1, [member], 'Data', iteration)
        for query in queries:
            D_sj_answer += [len(query(approximation_instance_delete.approximated_data))]
            D_sj_answer_golden_standard += [len(query(deleted_df))]
        answer = np.array(D_v_answer) - np.array(D_sj_answer)
        answer_golden_standard = np.array(D_v_answer_golden_standard) - np.array(D_sj_answer_golden_standard)
        return answer, answer_golden_standard
    # Maybe the golden standard should only be computed with in the timestamp?... Which means I
    # need to implement a functionality to compute the golden standard rather than mix this version.
    # But if the result is not good, we can compute the golden standard by summing the result in different
    # node up
    # Tomorrow you should complete the full version with re-start

    def answer_queries_golden_standard(self, nodes, cur_index, queries, member, epsilon=1, delta=0, iteration=500):
        for node in nodes:
            delete_df = pd.DataFrame(columns=self.config.keys())
            for index in range(cur_index + 1):
                if index == node.index:
                    node_df = node.df


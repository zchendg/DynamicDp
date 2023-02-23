import math

import numpy as np
import pandas as pd
import sys
import json
from mbi import Domain
from approximation_instance import ApproximationInstance
from node import Node
from ipp import IPP
from current_df import CurrentDf
from query import Query

# pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)


# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', 1000)

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


# Return the data set contains remaining items after the closet left-ancestor of v
# Need to deduce all the data contained in 'left-ancestors' of the current node
def df_after_closest_left_ancestor(node_list, index, current_df):
    set_diff_df = current_df.data
    for node in query_nodes(node_list, index):
        set_diff_df = pd.concat([set_diff_df, node.df, node.df]).drop_duplicates(keep=False)
    return set_diff_df


# Discrete raw data, insert empty data for timestamp padding
def sparse_data(data, frac=1, sparse_ratio=100):
    empty_data = pd.DataFrame(columns=data.columns, index=[i for i in range(sparse_ratio * len(df))])
    shuffle_data = pd.concat([data, empty_data]).sample(frac=frac).reset_index(drop=True)
    return shuffle_data


def insert_deletion_data(data, concentrate=True):
    if concentrate:
        data.insert(data.shape[1], 'update', np.nan)
        for i in range(len(data)):
            if data.loc[i].all():
                continue
            else:
                data.loc[i, 'update'] = 1
                data.loc[len(data)] = data.loc[i]
                data.loc[len(data) - 1, 'update'] = -1
    else:
        data.insert(data.shape[1], 'update', np.nan)
        for i in reversed(range(len(data))):
            if data.loc[i].all():
                continue
            else:
                data.loc[i, 'update'] = 1
                # The row index to be inserted
                row_index = np.random.randint(i, len(data))
                deletion_row = data.loc[i:i].copy()
                deletion_row.loc[i, 'update'] = -1
                data_part1 = data.loc[0:row_index]
                data_part2 = data.loc[row_index + 1:]
                data = pd.concat([data_part1, deletion_row, data_part2], ignore_index=True)
    return data


def query_nodes(node_list, index):
    nodes = [node_list[index]]
    if is_two_power(len(node_list) - 1):
        return nodes
    else:
        return nodes + query_nodes(
            node_list[0:find_far_left_ancestor_index(node_list[index].index, node_list[index].height) + 1],
            find_far_left_ancestor_index(node_list[index].index, node_list[index].height))


# This function unit the query's answer on nodes
def answer_queries(nodes, cur_index, queries, member, epsilon, delta=0):
    epsilon_budge = {node: 6 * epsilon / (np.square(np.pi * (node.height + 1))) for node in nodes}
    delta_budge = {node: 6 * delta / (np.square(np.pi * (node.height + 1))) for node in nodes}
    answer = [0] * len(queries)
    answer_ground_truth = [0] * len(queries)
    for node in nodes:
        queries_answer, queries_answer_ground_truth = answer_node_queries(node, nodes, cur_index, queries, member, epsilon_budge, delta_budge)
        answer = np.array(answer) + np.array(queries_answer)
        answer_ground_truth = np.array(answer_ground_truth) + np.array(queries_answer_ground_truth)
    return answer, answer_ground_truth


# Return a dataframe on the node
def answer_node_queries(node, node_list, cur_index, queries, member, epsilon, delta=0):
    # Initiate new object delete_df to store the data in the deletion-only problem
    delete_df = pd.DataFrame(columns=config.keys())
    # These variables store the query answer for the approximated dataset
    D_v_answer = []
    D_sj_answer = []
    # These variables store the query answer for the ground truth
    D_v_answer_ground_truth = []
    D_sj_answer_ground_truth = []
    for index in range(cur_index + 1):
        if index == node.index:
            node_df = node.df
            r = 1
            epsilon_r, delta_r = 3 * epsilon[node] / (2 * np.square(np.pi * r)), 2 * delta[node] / (
                        2 * np.square(np.pi * r))
            widetilde_n_v = len(node_df) + np.random.laplace(loc=0, scale=1 / epsilon_r)
            approximation_instance = ApproximationInstance(node_df, domain, 1, [member], 'Data', 500)
            for query in queries:
                D_v_answer += [len(query(approximation_instance.approximated_data.df))]
                D_v_answer_ground_truth += [len(query(node_df))]
        elif node.index < index <= cur_index:
            # Currently, I omit the restart procedure for simplicity
            delete_df = pd.concat(delete_df, node_list[index].delete_df).drop_duplicates(keep=False)
    # deleted_df refers to the contents of the database stored
    # in the current node remaining at time T after deletion
    deleted_df = pd.merge(delete_df, node.df)
    # The code is up to here
    approximation_instance_delete = ApproximationInstance(deleted_df, domain, 1, [member], 'Data', 500)
    for query in queries:
        D_sj_answer += [len(query(approximation_instance.approximated_data.df))]
        D_sj_answer_ground_truth += [len(query(deleted_df))]
    answer = np.array(D_v_answer) - np.array(D_sj_answer)
    answer_ground_truth = np.array(D_v_answer_ground_truth) - np.array(D_sj_answer_ground_truth)
    return answer, answer_ground_truth


def testing(node_list, ipp_instance, column_number=5, each_query_size=100, epsilon=1, delta=0):
    print('Testing start')
    query_instance = Query(config, column_number, each_query_size)
    # for each time stamp, we make some query test on it.
    for index in range(1, len(ipp_instance.get_segment())-1):
        querynodes = query_nodes(node_list[0: index + 1], index)
        print('At node with index %d, we implement the testing:' % index)
        for member in query_instance.queries.keys():
            answer, answer_ground_truth = answer_queries(querynodes, index, query_instance.queries[member], member, epsilon, delta)
            mse = ((np.array(answer) - np.array(answer_ground_truth))**2).mean()
            print("Testing on %s" % member, "Mean Square Error: %s" % mse)

# The order of the data is used as the insertion order, and the insertion of 1 to the original data means insertion,
# and each inserted data is deleted at some time node in the future
# We should suppose all the entries is not duplicated, then we can use the difference
# between left ancestor and current df
def main(argv):
    # Store the node list for the tree
    node_list = [Node(0, config.keys())]
    ipp_instance = IPP(df, 5, 0.05)
    print('ipp_instance:', ipp_instance)
    # For t in range(UPPERBOUND): When the segment is updated, we need to create a new node
    # Create a dataframe to store the data currently in the set
    cur_df = CurrentDf(config.keys())
    cur_deletion_df = CurrentDf(config.keys())
    for t in range(len(df)):
        ipp_instance.update_segment(t)
        if t == len(df) - 1:
            cur_df.current_df_update(df.iloc[[t]], t)
            ipp_instance.segment.append(t)
        if ipp_instance.segment[-1] == t:
            # First, create a new node, store the data in the new node
            # The new node is going to be created
            # We first insert the current df into the last node
            if len(ipp_instance.get_segment()) == 1:
                cur_df.current_df_update(df.iloc[[t]], t)
                continue
            elif is_two_power(len(ipp_instance.get_segment()) - 1):
                node_list.append(Node(len(node_list), config.keys()))
                node_list[-1].add_items(cur_df.data)
                node_list[-1].add_items(cur_deletion_df.data, delete_df=True)
                cur_deletion_df.renew()
            else:
                node_list.append(Node(len(node_list), config.keys()))
                # print('node_list[-1]', node_list[-1])
                node_list[-1].add_items(df_after_closest_left_ancestor(node_list, len(node_list) - 1, cur_df))
                node_list[-1].add_items(cur_deletion_df.data, delete_df=True)
                cur_deletion_df.renew()
        # For linear query, we need to keep track of the deletion time of the item
        cur_df.current_df_update(df.iloc[[t]], t)
        cur_deletion_df.add_deletion_item(df.iloc[[t]], t)
    print(node_list[1:])
    # print(ipp_instance.get_segment())
    testing(node_list, ipp_instance, 5, 10)
    print('Testing finished')


if __name__ == '__main__':
    domain = "/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/data/adult-domain.json"
    config = json.load(open(domain))
    domain = Domain(config.keys(), config.values())
    df = pd.read_csv('data/adult.csv', sep=',')
    df_title = df.columns
    df = df.iloc[0:100]
    df = sparse_data(df, 1, 10)
    df = insert_deletion_data(df, False)
    # print(df)
    UPPERBOUND = len(df)
    main(sys.argv)
    # Data-structure has been constructed, following is the query performance estimation
    # Outline:
    # 1. Generate queries list
    # 2. Query on the current dataset and the data union generated by the black-box algorithm
    # 3. Use Mean-Square Error to estimate performance
    # 4. Configuration optimization: potential parameter: black-box iteration loop, epsilon

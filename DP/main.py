import math

import numpy as np
import pandas as pd
import sys
import json
from node import Node
from ipp import IPP
from current_df import CurrentDf
from mbi import Domain
from approximation_instance import ApproximationInstance


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


def item_update(node, row):
    if row.x == 1:
        node.add_item(row)
    elif row.x == -1:
        node.delete_item(row)
    else:
        return


def query_nodes(node_list, index):
    nodes = [node_list[index]]
    if is_two_power(len(node_list) - 1):
        return nodes
    else:
        return nodes + query_nodes(
            node_list[0:find_far_left_ancestor_index(node_list[index].index, node_list[index].height) + 1],
            find_far_left_ancestor_index(node_list[index].index, node_list[index].height))


def answer_query(nodes, queries, segment, epsilon, delta=0):
    epsilon_budge = {node: 6 * epsilon / (np.square(np.pi * (node.height + 1))) for node in nodes}
    delta_budge = {node: 6 * delta / (np.square(np.pi * (node.height + 1))) for node in nodes}
    query_df = pd.DataFrame({'x': [], 'y': []})
    for node in nodes:
        answer_node_query(node, nodes, segment, epsilon_budge, delta_budge)


# Return a dataframe on the node
def answer_node_query(node, nodes, segment, epsilon, delta=0):
    # Initiate an new delete_df object to store the data in the deletion-only problem
    delete_df = pd.DataFrame({'x': [], 'y': []})
    for index in range(len(segment)):
        if index == node.index:
            dv = node.df
            r = 1
            epsilon_r, delta_r = 3 * epsilon / (2 * np.square(np.pi * r)), 2 * delta / (2 * np.square(np.pi * r))
            widetilde_n_v = len(dv) + np.random.laplace(loc=0, scale=1 / epsilon_r)
        elif node.index < index <= len(segment):
            # Currently, I omit the restart procedure for simplicity
            delete_df = pd.concat(delete_df, nodes[index].delete_df).drop_duplicates(keep=False)
    delete_df = pd.merge(delete_df, node.df)
    # The code is up to here
    instance = ApproximationInstance(delete_df, domain, 1, ['y'], 'Data')


# Return the data set contains remaining items after the closet left-ancestor of v
# Need to deduce all the data contained in 'left-ancestors' of the current node
def df_after_closest_left_ancestor(node_list, index, current_df):
    set_diff_df = current_df.data
    for node in query_nodes(node_list, index):
        set_diff_df = pd.concat([set_diff_df, node.df, node.df]).drop_duplicates(keep=False)
    return set_diff_df


# We should suppose all the entries is not duplicated, then we can use the difference
# between left ancestor and current df
def main(argv):
    # Store the node list for the tree
    node_list = [Node(0)]
    ipp_instance = IPP(df, 5, 0.05)
    print(ipp_instance)
    # For t in range(UPPERBOUND): When the segment is updated, we need to create a new node
    # Create a dataframe to store the data currently in the set
    cur_df = CurrentDf()
    cur_deletion_df = CurrentDf()
    for t in range(len(df)):
        ipp_instance.update_segment(t)
        if t == len(df) - 1:
            cur_df.current_df_update(df.loc[t])
            ipp_instance.segment.append(t)
        if ipp_instance.segment[-1] == t:
            # First, create a new node, store the data in the new node
            # The new node is going to be created
            # We first insert the current df into the last node
            if len(ipp_instance.get_segment()) == 1:
                cur_df.current_df_update(df.loc[t])
                continue
            elif is_two_power(len(ipp_instance.get_segment()) - 1):
                node_list.append(Node(len(node_list)))
                node_list[-1].add_items(cur_df.data)
                node_list[-1].add_items(cur_deletion_df.data, delete_df=True)
                cur_deletion_df.renew()
            else:
                node_list.append(Node(len(node_list)))
                # print('node_list[-1]', node_list[-1])
                node_list[-1].add_items(df_after_closest_left_ancestor(node_list, len(node_list) - 1, cur_df))
                node_list[-1].add_items(cur_deletion_df.data, delete_df=True)
                cur_deletion_df.renew()
        # For linear query, we need to keep track of the deletion time of the item
        cur_df.current_df_update(df.loc[t])
        cur_deletion_df.add_deletion_item(df.loc[t])
    print('-------------------')
    print('nodes:\n', node_list[1:])
    print(ipp_instance.get_segment())


if __name__ == '__main__':
    domain = "/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/bb_example.json"
    config = json.load(open(domain))
    domain = Domain(config.keys(), config.values())
    df = pd.read_csv('/Users/chenzijun/Library/CloudStorage/OneDrive-HKUSTConnect/Study/Program/DP/simple DP '
                     'dataset.csv', sep=',')
    # df = df.loc[0:1000]
    UPPERBOUND = len(df)
    main(sys.argv)

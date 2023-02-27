import numpy as np
import pandas as pd
from node import Node


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
    def __init__(self):
        self.node_list = []

    def insert_item(self, index, item):
        self.node_list[index - 1].add_item(item)

    def create_node(self, index):
        self.node_list += Node(index)

    def df_after_closest_left_ancestor(self, index, current_df):
        set_diff_df = current_df.data
        for node in self.query_nodes(self.node_list, index):
            set_diff_df = pd.concat([set_diff_df, node.df, node.df]).drop_duplicates(keep=False)
        return set_diff_df

    def query_nodes(self, node_list, index):
        nodes = [node_list[index]]
        if is_two_power(len(node_list) - 1):
            return nodes
        else:
            return nodes + self.query_nodes(
                node_list[0:find_far_left_ancestor_index(node_list[index].index, node_list[index].height) + 1],
                find_far_left_ancestor_index(node_list[index].index, node_list[index].height))

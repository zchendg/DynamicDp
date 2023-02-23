import numpy as np
import pandas as pd
from node import Node


class Tree:
    def __init__(self):
        self.node_list = []

    def insert_item(self, index, item):
        self.node_list[index - 1].add_item(item)

    def create_node(self, index):
        self.node_list += Node(index)

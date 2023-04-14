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

class Insertion_Deletion:
    def __init__(self, config):
        self.insertion_node_list = [Node(0, config.keys(), 0)]
        self.deletion_node_list = [Node(0, config.keys(), 0)]
        self.config = config
        self.domain = Domain(config.keys(), config.values())
        self.query_instance

    def insert_item(self, index, item):

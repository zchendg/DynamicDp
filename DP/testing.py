import numpy as np
import pandas as pd
import json
from approximation_instance import ApproximationInstance
from mbi import Domain, Dataset
from query import Query


class Testing:
    def __init__(self, config, dynamic_tree, ipp, column_number=1, each_query_size=100, epsion=1, delta=0):
        self.query_instance = Query(config, column_number, each_query_size)

    def testing(self):
        for index in range(1, len(ipp.instance.get_segment()))
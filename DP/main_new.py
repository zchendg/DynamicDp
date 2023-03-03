import numpy as np
import pandas as pd
import json
import logging
import os
import argparse
from mbi import Domain
import auxiliary
from dynamic_tree import Dynamic_Tree
from approximation_instance import ApproximationInstance
from node import Node
from ipp import IPP
from current_df import CurrentDf
from query import Query
from datetime import datetime

if not os.path.exists('./log'):
    os.mkdir('./log')
time_format = "%Y-b-%d_%H-%M-%S"

if 1:
    # ----Command Line Arguments Section--------
    parser = argparse.ArgumentParser()

    # General configuration
    parser.add_argument('--dataset_path', type=str, default='./data/adult.csv')
    parser.add_argument('--domain_path', type=str, default='./data/adult-domain.json')
    parser.add_argument('--data_size', type=int, default=1000)
    parser.add_argument('--sparse_ratio', type=int, default=10, help='The ratio between the Nan data and meaningful '
                                                                     'data')
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--iteration', type=int, default=500)

    # Command line arguments parser
    print('******** Parsing Parameter********')
    args = parser.parse_args()

    # Initialize the logger
    logging_file_name = './log/' + datetime.now().strftime(time_format) + '.log'
    logging.basicConfig(filename=logging_file_name, format='[%(asctime)s][%(levelname)s] - %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S %p", level=logging.INFO)
    logger = logging.getLogger()
    logger.info('Arguments: ' + str(args))
    # ------------------------------------------------------------------


def main():
    # ---- Initialization Section --------
    print('******** Initialization ********')
    logger.info('******** Initialization ********')
    # Loading dataset
    data = pd.read_csv(args.dataset_path, sep=',').iloc[0:args.data_size]
    data = auxiliary.sparse_data(data, 1, 10)
    data = auxiliary.insert_deletion_data(data, False)
    config = json.load(open(args.domain_path))
    domain = Domain(config.keys(), config.values())
    UPPERBOUND = len(data)
    logger.info('Data information: %s' % config)

    # ---- Construction Section --------
    dynamic_tree = Dynamic_Tree(config.keys())
    ipp_instance = IPP(data, args.epsilon, args.beta)
    # Dataframe that stores the current dataset
    cur_df = CurrentDf(config.keys())
    cur_deletion_df = CurrentDf(config.keys())

    # ---- Establish Dynamic Tree --------
    print('******** Create Dynamic Tree ********')
    logger.info('******** Create Dynamic Tree ********')
    for t in range(UPPERBOUND):
        ipp_instance.update_segment(t)
        if t == len(data) - 1:
            cur_df.current_df_update(data.iloc[[t]], t)
            ipp_instance.segment.append(t)
        if ipp_instance.segment[-1] == t:
            # First, create a new node, store the data in the new node
            # The new node is going to be created
            # We first insert the current df into the last node
            if len(ipp_instance.get_segment()) == 1:
                cur_df.current_df_update(data.iloc[[t]], t)
                continue
            elif auxiliary.is_two_power(len(ipp_instance.get_segment()) - 1):
                dynamic_tree.create_left_most_node(cur_df.data, cur_deletion_df.data)
                cur_deletion_df.renew()
            else:
                dynamic_tree.create_internal_node(cur_df.data, cur_deletion_df.data)
                cur_deletion_df.renew()





if __name__ == '__main__':
    domain = "./data/adult-domain.json"

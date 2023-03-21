import sys

import pandas as pd
import json
import logging
import os
import argparse
import auxiliary
import time
from tqdm import tqdm
from tqdm._tqdm import trange
from dynamic_tree import DynamicTree
from ipp import IPP
from current_df import CurrentDf
from datetime import datetime
from my_logger import Logger

if not os.path.exists('./log'):
    os.mkdir('./log')
time_format = "%Y-b-%d_%H-%M-%S"

if 1:
    # ----Command Line Arguments Section--------
    parser = argparse.ArgumentParser()

    # General configuration
    parser.add_argument('--dataset_path', type=str, default='./data/adult.csv')
    parser.add_argument('--domain_path', type=str, default='./data/adult-domain.json')
    parser.add_argument('--data_size', type=int, default=100)
    parser.add_argument('--sparse_ratio', type=int, default=10, help='The ratio between the Nan data and meaningful '
                                                                     'data')
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--iteration', type=int, default=500)

    # Command line arguments parser
    print('******** Parsing Parameter********')
    args = parser.parse_args()

    # Initialize the logger
    logger_file_name = './log/' + datetime.now().strftime(time_format) + '.log'
    logger = Logger(logger_file_name, sys.stdout)
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
    logger.info('Data sparsification complete')
    config = json.load(open(args.domain_path))
    UPPERBOUND = len(data)
    logger.info('Data information: %s' % config)

    # ---- Construction Section --------
    dynamic_tree = DynamicTree(config)
    ipp_instance = IPP(data, args.epsilon, args.beta)
    # ipp_instance = IPP(data, args.epsilon, args.beta)
    # Dataframe that stores the current dataset
    cur_data = CurrentDf(config.keys())
    cur_deletion_data = CurrentDf(config.keys())

    # ---- Establish Dynamic Tree --------
    print('******** Create Dynamic Tree ********')
    logger.info('******** Create Dynamic Tree ********')
    for t in trange(UPPERBOUND):
        ipp_instance.update_segment(t)
        if t == (UPPERBOUND - 1):
            cur_data.current_df_update(data.iloc[[t]], t)
            cur_deletion_data.add_deletion_item(data.iloc[[t]], t)
            ipp_instance.segment.append(t)
        if ipp_instance.segment[-1] == t:
            # First, create a new node, store the data in the new node
            # The new node is going to be created
            # We first insert the current df into the last node
            if len(ipp_instance.get_segment()) == 1:
                cur_data.current_df_update(data.iloc[[t]], t)
                continue
            elif auxiliary.is_two_power(len(ipp_instance.get_segment()) - 1):
                dynamic_tree.create_leftmost_node(cur_data, cur_deletion_data, t)
                cur_deletion_data.renew()
            else:
                dynamic_tree.create_internal_node(cur_data, cur_deletion_data, t)
                cur_deletion_data.renew()
        # For linear query, we need to keep track of the deletion time of the item
        cur_data.current_df_update(data.iloc[[t]], t)
        cur_deletion_data.add_deletion_item(data.iloc[[t]], t)
    logger.info('The dynamic interval tree consists of nodes: %s' % dynamic_tree.node_list[1:])
    logger.info('Infinite Private Partitioning: %s' % ipp_instance)
    print('******** Testing Started ********')
    logger.info('******** Testing Started ********')
    # Modification
    # dynamic_tree.node_list[4].df = pd.read_csv(args.dataset_path, sep=',').iloc[0:10000]
    # Modification
    # dynamic_tree.testing_index(3, epsilon=args.epsilon, iteration=args.iteration, logger=logger)
    dynamic_tree.testing(ipp_instance, 1, 100, args.epsilon, args.delta, args.beta, args.iteration, logger)
    print('******** Testing Finished ********')
    logger.info('******** Testing Finished ********')
    return -1


if __name__ == '__main__':
    main()

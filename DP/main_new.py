import sys
import pandas as pd
import json
import logging
import os
import argparse
import auxiliary
import auxiliary1
import analysis_data_1
import time
from tqdm import tqdm
from data_loader import DataLoader
from tqdm._tqdm import trange
from dynamic_tree import DynamicTree
from insertion_deletion import InsertionDeletionMechanism
from ipp import IPP
from query import Query
from current_df import CurrentDf
from datetime import datetime
from my_logger import Logger

# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# pd.set_option('max_colwidth', 100)

if not os.path.exists('./result'):
    os.mkdir('./result')
time_format = "%Y-b-%d_%H-%M-%S"
result_path = './result/' + datetime.now().strftime(time_format)
os.mkdir(result_path)
logger_file_name = result_path + '/data.log'
figure_file_name = result_path + '/figure_'

if 1:
    # ----Command Line Arguments Section--------
    parser = argparse.ArgumentParser()

    # General configuration
    parser.add_argument('--dataset_path', type=str, default='./data/adult.csv')
    parser.add_argument('--domain_path', type=str, default='./data/adult-domain.json')
    parser.add_argument('--data_size', type=int, default=100)
    parser.add_argument('--sparse_ratio', type=int, default=10, help='The ratio between the Nan data and meaningful '
                                                                     'data')
    parser.add_argument('--dynamic_size', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--iteration', type=int, default=500)
    parser.add_argument('--column_number', type=int, default=1)
    parser.add_argument('--query_size', type=int, default=10)

    # Command line arguments parser
    print('******** Parsing Parameter********')
    args = parser.parse_args()

    # Initialize the logger
    # logger_file_name = './log/' + datetime.now().strftime(time_format) + '.log'
    logger = Logger(logger_file_name, sys.stdout)
    # logging.basicConfig(filemode='w', filename=logger_file_name, level=logging.INFO)
    # logger = logging.getLogger()
    logger.info('Arguments: ' + str(args))
    # ------------------------------------------------------------------


def main():

    # ---- Initialization Section --------
    print('******** Initialization ********')
    logger.info('******** Initialization ********')
    # Loading dataset
    data = pd.read_csv(args.dataset_path, sep=',').iloc[0:args.data_size]
    # data = auxiliary.process_data(data, sparse=False)
    # data = auxiliary.generate_fixed_size_data(data, args.dynamic_size)
    data_loader_instance = DataLoader('fixed size', data, args.dynamic_size, logger=logger)
    data = data_loader_instance.dynamic_data
    insertion_only_data = data_loader_instance.insertion_only_data
    deletion_only_data = data_loader_instance.deletion_only_data
    # data = auxiliary.sparse_data(data, 1, 10)
    # data = auxiliary.insert_deletion_data(data, False)
    config = json.load(open(args.domain_path))
    UPPERBOUND = len(data)
    logger.info('Data information: %s' % config)
    # ---- Construction Section --------
    query_instance = Query(config, query_type='linear query', random_query=True, query_size=args.query_size, logger=logger)
    dynamic_tree = DynamicTree(config, query_instance)
    insertion_deletion_instance = InsertionDeletionMechanism(config, query_instance)
    ipp_instance = IPP(data, args.epsilon, args.beta)
    # Dataframe that stores the current dataset
    # cur_data: current data upto the timestamp t
    # cur_deletion_data: deletion data between two nodes
    # cur_deleted_data: treating deletion as insertion
    cur_data = CurrentDf(config.keys())
    cur_deletion_data = CurrentDf(config.keys())
    cur_inserted_data = CurrentDf(config.keys())
    cur_deleted_data = CurrentDf(config.keys())

    # ---- Establish Dynamic Tree --------
    print('******** Create Dynamic Tree Starts********')
    logger.info('******** Create Dynamic Tree Starts********')
    for t in trange(UPPERBOUND):
        ipp_instance.update_segment(t)
        if t == (UPPERBOUND - 1):
            cur_data.current_df_update(data.iloc[[t]], t)
            cur_deletion_data.add_deletion_item(data.iloc[[t]], t)
            # cur_data_deleted.add_deletion_item(data.iloc[[t]], t)
            ipp_instance.segment.append(t)
        if ipp_instance.segment[-1] == t:
            # First, create a new node, store the data in the new node
            # The new node is going to be created
            # We first insert the current df into the last node
            if len(ipp_instance.get_segment()) == 1:
                cur_data.current_df_update(data.iloc[[t]], t)
                cur_inserted_data.add_insertion_item(data.iloc[[t]], t)
                cur_deleted_data.add_deletion_item(data.iloc[[t]], t)
                continue
            elif auxiliary.is_two_power(len(ipp_instance.get_segment()) - 1):
                # Create leftmost node
                dynamic_tree.create_leftmost_node(cur_data, cur_deletion_data, t)
                insertion_deletion_instance.insertion_tree.create_leftmost_node(cur_inserted_data, t)
                insertion_deletion_instance.deletion_tree.create_leftmost_node(cur_deleted_data, t)
                cur_deletion_data.renew()
            else:
                dynamic_tree.create_internal_node(cur_data, cur_deletion_data, t)
                insertion_deletion_instance.insertion_tree.create_internal_node(cur_inserted_data, t)
                insertion_deletion_instance.deletion_tree.create_internal_node(cur_deleted_data, t)
                cur_deletion_data.renew()
        # For linear query, we need to keep track of the deletion time of the item
        cur_data.current_df_update(data.iloc[[t]], t)
        cur_deletion_data.add_deletion_item(data.iloc[[t]], t)
        cur_inserted_data.add_insertion_item(data.iloc[[t]], t)
        cur_deleted_data.add_deletion_item(data.iloc[[t]], t)
    print('******** Create Dynamic Tree Finishes********')
    logger.info('******** Create Dynamic Tree Finishes********')
    logger.info('Infinite Private Partitioning: %s' % ipp_instance)
    logger.info('The dynamic interval tree consists of nodes: %s' % dynamic_tree.node_list[1:])
    logger.info('The insertion tree consists of nodes: %s ' % insertion_deletion_instance.insertion_tree.node_list[1:])
    logger.info('The deletion tree consists of nodes: %s ' % insertion_deletion_instance.deletion_tree.node_list[1:])
    print('******** Testing Started ********')
    logger.info('******** Testing Started ********')
    # Modification
    # dynamic_tree.node_list[4].df = pd.read_csv(args.dataset_path, sep=',').iloc[0:10000]
    # Modification
    # dynamic_tree.testing_index(8, epsilon=args.epsilon, iteration=args.iteration, logger=logger)
    dynamic_tree.testing(ipp_instance, args.epsilon, args.delta, args.beta, args.iteration, logger)
    insertion_deletion_instance.testing(ipp_instance, args.epsilon, args.delta, args.beta, args.iteration, logger)
    auxiliary1.store_answer(dynamic_tree, insertion_deletion_instance, ipp_instance, logger=logger)
    print('******** Testing Finished ********')
    logger.info('******** Testing Finished ********')
    logger.info('******** Drawing Figure started ********')
    # dynamic_tree.draw_diagram(ipp_instance, figure_file_name)
    analysis_data_1.draw_mean_diagram(dynamic_tree, insertion_deletion_instance, query_instance, ipp_instance, figure_file_name)
    analysis_data_1.draw_error_diagram(dynamic_tree, insertion_deletion_instance, query_instance, ipp_instance, figure_file_name)
    logger.info('******** Drawing Figure finished ********')
    print('******** Results stored in %s ********' % result_path)
    return -1


if __name__ == '__main__':
    main()

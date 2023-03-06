import os
import pandas as pd
import json
import my_logging
import argparse
import auxiliary
from datetime import datetime
from dynamic_tree import DynamicTree


if not os.path.exists('./log'):
    os.mkdir('./log')
time_format = "%Y-b-%d_%H-%M-%S"

if 1:
    # Command line arguments parser
    parser = argparse.ArgumentParser()

    # General configuration
    parser.add_argument('--dataset_path', type=str, default='./data/adult.csv')
    parser.add_argument('--domain_path', type=str, default='./data/adult-domain.json')
    parser.add_argument('--data_size', type=int, default=100)
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
    UPPERBOUND = len(data)
    logger.info('Data information: %s' % config)

    # ---- Construction Section --------


    return -1


if __name__ == '__main__':
    main()

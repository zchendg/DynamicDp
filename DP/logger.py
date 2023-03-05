import os
import logging
from datetime import datetime

if not os.path.exists('./log'):
    os.mkdir('./log')
time_format = "%Y-b-%d_%H-%M-%S"

if 1:
    # Command line arguments parser

    # Initialize the logger
    logging_file_name = './log/' + datetime.now().strftime(time_format) + '.log'
    logging.basicConfig(filename=logging_file_name, datefmt="%m/%d/%Y %H:%M:%S %p", level=logging.INFO)
    logger = logging.getLogger()
    logger.info('Arguments: test 1')
    # ------------------------------------------------------------------


def main():
    logger.info('finished')
    return -1


if __name__ == '__main__':
    main()

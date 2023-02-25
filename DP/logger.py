import os
import logging
from datetime import datetime


if 1:
    if not os.path.exists('./log'):
        os.mkdir('./log')
    time_format = "%Y-%b-%d_%H-%M-%S"
    # Initialize the logger
    logging_file_name = './log/' + datetime.now().strftime(time_format) + '.log'
    logging.basicConfig(filename=logging_file_name,
                        format='[%(asctime)s][%(levelname)s] - %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S %p",
                        level=logging.INFO)

    # logging.info("Arguments: ")


def main():
    logging.debug('This is debug message')
    logging.info('This is info message')
    logging.warning('This is warning message')
    print('success')
    return


if __name__ == '__main__':
    # time_format = "%Y-%b-%d_%H-%M-%S"
    # logging_file_name = './log/' + datetime.now().strftime(time_format) + '.log'
    # logging.basicConfig(filename=logging_file_name,
    #                     format='[%(asctime)s][%(levelname)s] - %(message)s',
    #                     datefmt="%m/%d/%Y %H:%M:%S %p",
    #                     level=logging.INFO,
    #                     filemode='w')
    main()

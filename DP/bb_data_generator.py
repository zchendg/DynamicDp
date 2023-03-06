import logging

from my_logger import Logger
from datetime import datetime
import sys

time_format = "%Y-b-%d_%H-%M-%S"
logger_file_name = './log/' + datetime.now().strftime(time_format) + '.log'
logger = Logger(logger_file_name, sys.stdout)
logger.info('Arguments:')
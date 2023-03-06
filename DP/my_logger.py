import sys
import datetime


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout, format='%Y-%m-%d %H:%M:%S'):
        self.terminal = stream
        self.log = open(filename, 'w')
        self.format = format

    def info(self, message):
        # self.terminal.write(message + '\n')
        prefix = str('[' + datetime.datetime.now().strftime(self.format) + ']' + ' - ')
        self.log.write(prefix + message + '\n')

    def flush(self):
        pass

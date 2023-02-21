import logging
import os.path
import time


class DataLogger(object):
    def __init__(self, log_file, logger=None):
        self.log_file = log_file
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(self.log_file, 'a', encoding='utf-8')
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '[%(asctime)s] %(funcName)s()--[%(levelname)s]: %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        fh.close()
        ch.close()

    def getlog(self):
        return self.logger

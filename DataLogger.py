import logging
import os.path
import colorlog


log_colors_config = {
    # 终端输出日志颜色配置
    # 'DEBUG': 'white',
    # 'INFO': 'cyan',
    # 'WARNING': 'yellow',
    # 'ERROR': 'red',
    # 'CRITICAL': 'bold_red',
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red,bg_white',
}

default_formats = {
    # 终端输出格式
    'color_format': '%(log_color)s%(asctime)s %(filename)s[line:%(lineno)d]--%(levelname)s: %(message)s',
    # 日志输出格式
    'log_format': '%(asctime)s %(filename)s[line:%(lineno)d]--%(levelname)s: %(message)s'
}


class DataLogger(object):
    """
    日志工具类
    :param log_file: 保存日志的文件路径
    :param logger: 指定logger
    """
    def __init__(self, log_file, logger=None):
        self.log_file = log_file
        self.logger = logging.getLogger(logger)
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.log_file, 'a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # formatter = logging.Formatter('[%(asctime)s] %(funcName)s()--[%(levelname)s]: %(message)s')
        formatter = logging.Formatter(default_formats.get('log_format'))
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)
        console_handler.setFormatter(colorlog.ColoredFormatter(fmt=default_formats.get('color_format'),
                                                               datefmt='%Y-%m-%d  %H:%M:%S',
                                                               log_colors=log_colors_config))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        file_handler.close()
        console_handler.close()

    def getlog(self):
        return self.logger

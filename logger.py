import logging

class Logger:

    def __init__(self):

        self.level = logging.ERROR

    @staticmethod
    def get_logger(unique_name, debug):

        logger = logging.getLogger(unique_name)

        if debug:
            level = logging.DEBUG

        else:
            level = logging.INFO

        logger.setLevel(level)

        hdr = logging.StreamHandler()

        formatter = logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s: %(message)s')

        hdr.setFormatter(formatter)

        logger.addHandler(hdr)

        return logger



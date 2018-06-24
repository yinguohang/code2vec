import logging
import os
import platform
import sys
import time

import tensorflow as tf
from tensorflow.python.framework.errors_impl import UnimplementedError, NotFoundError


def get_optimizer(name):
    if name == "adadelta":
        return tf.train.AdadeltaOptimizer(1.0)
    if name == "adam":
        return tf.train.AdamOptimizer(0.001)


def detect_platform():
    is_pai = True
    try:
        tf.gfile.GFile("oss://file_not_existed", "r").read()
    except UnimplementedError:
        is_pai = False
    except NotFoundError:
        pass
    return 'PAI' if is_pai else platform.system().upper()


def init_tf_logging(log_path):
    """
    Redirect tf.logging to stdout and also a separate log file
    :param log_path: Base directory for log files
    """
    # Create log formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Custom GFile Handler
    class GFileHandler(logging.FileHandler):
        def __init__(self, filename, mode='a', encoding=None, delay=False):
            self.filename = filename
            logging.FileHandler.__init__(self, filename, mode, encoding, delay)

        def _open(self):
            return tf.gfile.Open(self.filename, self.mode)

    # Create log directory if not existed
    if not tf.gfile.Exists(log_path):
        tf.gfile.MkDir(log_path)

    # Create file handler
    fh = GFileHandler(os.path.join(log_path, "tensorflow-"
                                   + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log"), mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # Init logging
    tf.logging._logger.removeHandler(tf.logging._handler)
    tf.logging._logger.addHandler(ch)
    tf.logging._logger.addHandler(fh)
    tf.logging._logger.setLevel(logging.DEBUG)

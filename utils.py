import logging
import platform

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


def create_file_handler(filename):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    return fh

import hashlib
import logging
import tempfile

import numpy as np
import os
import platform
import sys
import time

import tensorflow as tf
from tensorflow.python.framework.errors_impl import UnimplementedError, NotFoundError

from numpy.compat import (basestring, is_pathlib_path)


def get_optimizer(optimizer, learning_rate):
    if optimizer == "adadelta":
        return tf.train.AdadeltaOptimizer(learning_rate)
    if optimizer == "adam":
        return tf.train.AdamOptimizer(learning_rate)
    if optimizer == "gd":
        return tf.train.GradientDescentOptimizer(learning_rate)
    raise RuntimeError("Unknown optimizer " + optimizer)


def detect_platform():
    is_pai = True
    try:
        tf.gfile.GFile("oss://file_not_existed", "r").read()
    except UnimplementedError:
        is_pai = False
    except NotFoundError:
        pass
    return 'PAI' if is_pai else platform.system().upper()


def file_checksum(filename):
    """
    Compute md5 checksum of the file with given name

    :param filename:  Name of the file
    :return:          Md5 checksum of the file
    """
    hash_md5 = hashlib.md5()
    with tf.gfile.Open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
        f.close()
    return hash_md5.hexdigest()


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
    logger = logging.getLogger('tensorflow')
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


def np_savez_compressed(file, *args, **kwds):
    """
    A wrapper for np.savez_compressed() to make it work with tf.gfile
    """
    if isinstance(file, basestring):
        if not file.endswith('.npz'):
            file = file + '.npz'
    elif is_pathlib_path(file):
        if not file.name.endswith('.npz'):
            file = file.parent / (file.name + '.npz')
    else:
        raise RuntimeError("Please specify filename in string format")

    zip_fd, zip_tempfile = tempfile.mkstemp(suffix='.npz')
    np.savez_compressed(zip_tempfile, *args, **kwds)

    tf.gfile.MakeDirs(os.path.dirname(file))
    tf.gfile.Copy(zip_tempfile, file, overwrite=True)
    os.close(zip_fd)

from os import path

import tensorflow as tf

from . import utils

base_path = {
    'PAI': 'oss://apsalgo-hz/force/codequailty/code2vec',
    'DARWIN': '/Users/jiangjunfang/Desktop/code2vec',
    'WINDOWS': ''
}

data_set = {
    'PAI': 'paths-18728',
    'DARWIN': 'paths-1000',
    'WINDOWS': 'paths-1000'
}


def init():
    flags = tf.app.flags

    flags.DEFINE_string("data_set", data_set[utils.detect_platform()], "Name of the data set to be used")

    flags.DEFINE_string("data_path", path.join(base_path[utils.detect_platform()], 'data'), "Absolute path of data directory")

    flags.DEFINE_string("log_path", path.join(base_path[utils.detect_platform()], 'log'), "Absolute path of log directory")

    flags.DEFINE_integer("context_bag_size", 100, "The number of context paths in AST to be used in training")

    flags.DEFINE_integer("node_embedding_size", 50, "Node (start and end) embedding size")

    flags.DEFINE_integer("path_embedding_size", 50, "Path embedding size")

    flags.DEFINE_string("optimizer", "adam", "Selected optimizer")

    flags.DEFINE_integer("FC1", 50, "FC1 size")

    flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")

    flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

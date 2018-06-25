import os
import tensorflow as tf

from . import utils

base_path = {
    'PAI': 'oss://apsalgo-hz/force/codequailty/code2vec',
    'DARWIN': '/Users/jiangjunfang/Desktop/code2vec',
    'WINDOWS': 'D:\\ml\\code2vec_tf'
}

data_set = {
    'PAI': 'paths-18728',
    'DARWIN': 'paths-1000',
    'WINDOWS': 'paths-1000'
}


def init():
    flags = tf.app.flags

    flags.DEFINE_string("data_set", data_set[utils.detect_platform()],
                        "Name of the data set to be used")

    flags.DEFINE_string("data_path", os.path.join(base_path[utils.detect_platform()], 'data'),
                        "Absolute path of data directory")

    flags.DEFINE_string("log_path", os.path.join(base_path[utils.detect_platform()], 'log'),
                        "Absolute path of log directory")

    flags.DEFINE_integer("context_bag_size", 100,
                         "The number of context paths in AST to be used in training")

    flags.DEFINE_integer("node_embedding_size", 100,
                         "Node (start and end) embedding size")

    flags.DEFINE_integer("path_embedding_size", 250,
                         "Path embedding size")

    flags.DEFINE_integer("encode_size", 150, "Context encoding size")

    flags.DEFINE_integer("classification", -1, "Number of class for classification, or use regression if less than 1")

    flags.DEFINE_integer("attention_layer_dimension", 30, "Dimension of attention layer")

    flags.DEFINE_float("dropout_rate", 0.5, "Dropout rate")

    flags.DEFINE_float("encoding_layer_penalty_rate", 0.003, "Encoding layer penalty rate")

    flags.DEFINE_float("attention_layer_penalty_rate", 0.2, "Attention layer penalty rate")

    flags.DEFINE_float("regression_layer_penalty_rate", 0.5, "Regression layer penalty rate")

    flags.DEFINE_string("optimizer", "adam", "Selected optimizer")

    if flags.FLAGS.optimizer == "adam":
        flags.DEFINE_string("learning_rate", 0.00005, "Learning rate")
    elif flags.FLAGS.optimizer == "adadelta":
        flags.DEFINE_string("learning_rate", 1.0, "Learning rate")

    flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")

    flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

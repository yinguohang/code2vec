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

    ############################
    # Global
    ############################

    # System Settings
    flags.DEFINE_string("data_set", data_set[utils.detect_platform()],
                        "Name of the data set to be used")

    flags.DEFINE_string("data_path", os.path.join(base_path[utils.detect_platform()], 'data/inner-class'),
                        "Absolute path of data directory")

    flags.DEFINE_string("log_path", os.path.join(base_path[utils.detect_platform()], 'log'),
                        "Absolute path of log directory")

    # Model Settings
    flags.DEFINE_string("optimizer", "adam", "Selected optimizer")

    if flags.FLAGS.optimizer == "adam":
        flags.DEFINE_string("learning_rate", 0.0001, "Learning rate")
    elif flags.FLAGS.optimizer == "adadelta":
        flags.DEFINE_string("learning_rate", 1.0, "Learning rate")

    flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")

    flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    flags.DEFINE_float("dropout_rate", 0.5, "Dropout rate")

    flags.DEFINE_integer("classification", -1,
                         "Number of class for classification, or use regression if less than 1")

    ############################
    # Embedding Layer
    ############################

    # Embedding Layer Structure Parameter
    flags.DEFINE_integer("embedding_bag_size", 100,
                         "The number of context paths in AST to be used in training")

    flags.DEFINE_integer("embedding_node_size", 150,
                         "Node (start and end) embedding size")

    flags.DEFINE_integer("embedding_path_size", 250,
                         "Path embedding size")

    ############################
    # Encoding Layer
    ############################

    # Encoding Layer Structure Parameter
    flags.DEFINE_integer("encoding_size", 150, "Context encoding size")

    # Encoding Layer Penalty Parameter
    flags.DEFINE_float("encoding_weight_penalty_rate", 0.01, "Encoding layer penalty rate")

    ############################
    # Attention Layer
    ############################

    # Attention Layer Structure Parameter
    flags.DEFINE_integer("attention_dimension_size", 30, "Dimension of attention layer")

    # Attention Layer Penalty Parameter
    flags.DEFINE_float("attention_weight_penalty_rate", 0.5, "Attention layer penalty rate")

    ############################
    # Regression Layer
    ############################

    # Regression Layer Structure Parameter
    flags.DEFINE_integer("regression_concat_vec_size", 100, "vector size when concatenating")

    flags.DEFINE_integer("regression_concat_feature_size", 1, "feature size when concatenating")

    flags.DEFINE_integer("regression_hidden_layer_size", 250, "Hidden units between concat layer and output layer")

    # Regression Layer Penalty Parameter
    flags.DEFINE_float("regression_vec_weight_penalty_rate", 0.03, "")

    flags.DEFINE_float("regression_feature_weight_penalty_rate", 0.1, "")

    flags.DEFINE_float("regression_layer_penalty_rate", 0.03, "Regression layer penalty rate")

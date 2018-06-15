# -*- coding:utf-8 -*-

# 主程序入口
import logging
import os
import time

import numpy as np
import tensorflow as tf

import utils
from data import DataReader
from model import Code2VecModel

tf.logging._logger.addHandler(utils.create_file_handler("tensorflow.log"))
tf.logging._logger.setLevel(logging.DEBUG)

data_path = {
    'PAI': 'oss://apsalgo-hz/force/codequailty/code2vec/data',
    'DARWIN': '/Users/jiangjunfang/Desktop/code2vec/data',
    'WINDOWS': ''
}

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_set",
                    default_value="paths-1000",
                    docstring='Name of the data set to be used')

flags.DEFINE_string("os_type",
                    default_value=utils.detect_platform(),
                    docstring="Current OS platform (PAI/WINDOWS/DARWIN)")

flags.DEFINE_string("data_path",
                    default_value=data_path[utils.detect_platform()],
                    docstring="Absolute path of data directory on current platform")

flags.DEFINE_integer("context_bag_size",
                     default_value=100,
                     docstring="The number of context paths in AST to be used in training")

flags.DEFINE_integer("node_embedding_size",
                     default_value=50,
                     docstring="Node (start and end) embedding size")

flags.DEFINE_integer("path_embedding_size",
                     default_value=50,
                     docstring="Path embedding size")

flags.DEFINE_boolean("allow_soft_placement",
                     default_value=True,
                     docstring="Allow device soft device placement")

flags.DEFINE_boolean("log_device_placement",
                     default_value=False,
                     docstring="Log placement of ops on devices")

flags.DEFINE_string("optimizer", "adam", "Selected optimizer")


flags.DEFINE_integer("FC1", 50, "FC1 size")


class Option:
    def __init__(self, reader):
        self.training = True
        self.node_embedding_size = FLAGS.node_embedding_size
        self.path_embedding_size = FLAGS.path_embedding_size
        self.encode_size = FLAGS.FC1
        self.node_cnt = reader.node_converter.cnt + 1
        self.path_cnt = reader.path_converter.cnt + 1


def train():
    reader = DataReader(os.path.join(FLAGS.data_path, FLAGS.data_set), FLAGS.context_bag_size)
    train_dataset = reader.train_dataset
    dev_dataset = reader.dev_dataset
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    eval_init_op = iterator.make_initializer(dev_dataset)
    batch_datas = iterator.get_next()
    start = batch_datas['start']
    path = batch_datas['path']
    end = batch_datas['end']
    score = batch_datas['score']

    with tf.variable_scope("model"):
        opt = Option(reader)
        lr = tf.placeholder(dtype=tf.float32, name='lr')
        train_model = Code2VecModel(start, path, end, score, opt)
        train_op = utils.get_optimizer(FLAGS.optimizer).minimize(train_model.loss)

    with tf.variable_scope('model', reuse=True):
        eval_opt = Option(reader)
        eval_opt.training = False
        eval_model = Code2VecModel(start, path, end, score, eval_opt)

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_init_op)
        for i in range(10):
            start_time = time.time()
            sum_loss = 0
            cnt = 0
            sess.run(train_init_op)
            while True:
                try:
                    _, loss_value, data = sess.run([train_op, train_model.loss, batch_datas])
                    sum_loss = sum_loss + loss_value * len(data["score"])
                    cnt = cnt + len(data["score"])
                except tf.errors.OutOfRangeError:
                    break
            eval_loss = eval(sess, eval_model, batch_datas, eval_init_op)
            tf.logging.info('Epoch %d: train-loss: %.8f, val-loss: %.8f, cost-time: %.4f s' %(
                i + 1, sum_loss / cnt, eval_loss, time.time() - start_time))


def eval(sess, model, batch_datas, test_init_op):
    scores = batch_datas['score']
    sess.run(test_init_op)
    sum_loss = 0
    cnt = 0
    while True:
        try:
            scores_rlt, loss_value = sess.run([scores, model.loss])
            sum_loss = loss_value * len(scores_rlt) + sum_loss
            cnt = cnt + len(scores_rlt)
        except tf.errors.OutOfRangeError:
            break
    return sum_loss / cnt


def main(_):
    tf.logging.info(str(FLAGS.__flags))
    train()


if __name__ == "__main__":
    np.random.seed(123)
    tf.set_random_seed(123)
    tf.app.run()

# -*- coding:utf-8 -*-

# 主程序入口
import os
import time

try:
    from Queue import PriorityQueue
except ImportError:
    from queue import PriorityQueue

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from data import DataReader
from model import Code2VecModel
from utils import config
from utils import utils

FLAGS = tf.app.flags.FLAGS

config.init()
utils.init_tf_logging(FLAGS.log_path)


class Option:
    def __init__(self, reader, training=True):
        self.training = training
        self.dropout_rate = FLAGS.dropout_rate
        self.classification = FLAGS.classification

        self.node_cnt = reader.node_converter.cnt + 1
        self.path_cnt = reader.path_converter.cnt + 1
        self.embedding_bag_size = FLAGS.embedding_bag_size
        self.embedding_node_size = FLAGS.embedding_node_size
        self.embedding_path_size = FLAGS.embedding_path_size

        self.encoding_size = FLAGS.encoding_size
        self.encoding_weight_penalty_rate = FLAGS.encoding_weight_penalty_rate

        self.attention_dimension_size = FLAGS.attention_dimension_size
        self.attention_weight_penalty_rate = FLAGS.attention_weight_penalty_rate

        self.regression_concat_vec_size = FLAGS.regression_concat_vec_size
        self.regression_concat_feature_size = FLAGS.regression_concat_feature_size
        self.regression_hidden_layer_size = FLAGS.regression_hidden_layer_size
        self.regression_vec_weight_penalty_rate = FLAGS.regression_vec_weight_penalty_rate
        self.regression_feature_weight_penalty_rate = FLAGS.regression_feature_weight_penalty_rate
        self.regression_layer_penalty_rate = FLAGS.regression_layer_penalty_rate


def train():
    reader = DataReader(os.path.join(FLAGS.data_path, FLAGS.data_set), FLAGS.embedding_bag_size)

    train_data = reader.train_dataset
    eval_data = reader.dev_dataset

    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

    batch_data = iterator.get_next()
    start = batch_data['start']
    path = batch_data['path']
    end = batch_data['end']
    score = batch_data['score']
    original_features = batch_data['original_features']

    train_init_op = iterator.make_initializer(train_data)
    eval_init_op = iterator.make_initializer(eval_data)

    with tf.variable_scope("code2vec_model"):
        opt = Option(reader)
        train_model = Code2VecModel(start, path, end, score, original_features, opt)
        train_op = utils.get_optimizer(FLAGS.optimizer, FLAGS.learning_rate).minimize(train_model.loss)

    with tf.variable_scope('code2vec_model', reuse=True):
        eval_opt = Option(reader, training=False)
        eval_model = Code2VecModel(start, path, end, score, original_features, eval_opt)

    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        min_eval_loss = PriorityQueue(maxsize=3)
        stable_min_loss = 0

        for i in range(1000):
            start_time = time.time()

            train_loss, train_acc = evaluate(sess, train_model, batch_data, train_init_op, train_op)
            eval_loss, eval_acc = evaluate(sess, eval_model, batch_data, eval_init_op)
            eval_reg_loss, eval_reg_acc = evaluate(sess, train_model, batch_data, eval_init_op)

            if not min_eval_loss.full():
                min_eval_loss.put(-eval_loss)
                stable_min_loss = 0
            else:
                k = min_eval_loss.get()
                if k >= -eval_loss:
                    stable_min_loss += 1
                else:
                    stable_min_loss = 0
                min_eval_loss.put(max(k, -eval_loss))

            if opt.classification > 0:
                tf.logging.info(
                    'Epoch %2d: train-loss: %.5f (acc=%.2f), val-loss: %.5f (acc=%.2f), min-loss: %.5f, cost: %.4f s'
                    % (i + 1, train_loss, train_acc, eval_loss, eval_acc, float(-np.mean(min_eval_loss.queue)),
                       time.time() - start_time))
            else:
                tf.logging.info(
                    'Epoch %2d: train-loss: %.5f, val-reg: %.5f, val-loss: %.5f, min-loss: %.5f, cost: %.4f s'
                    % (i + 1, train_loss, eval_reg_loss, eval_loss, float(-np.mean(min_eval_loss.queue)),
                       time.time() - start_time))

            if stable_min_loss >= 5 and i > 50: break


def evaluate(sess, model, batch_data, batch_init_op, op=None):
    sess.run(batch_init_op)
    batch_loss_lst = []
    batch_acc_lst = []

    while True:
        try:
            if op is not None:
                _, loss_value, data, acc = sess.run([op, model.loss, batch_data, model.acc])
            else:
                loss_value, data, acc = sess.run([model.loss, batch_data, model.acc])
            batch_acc_lst.append((acc[0], len(data['score'])))
            batch_loss_lst.append((loss_value, len(data['score'])))
        except tf.errors.OutOfRangeError:
            break

    loss = sum(l * n for l, n in batch_loss_lst) / sum(n for _, n in batch_loss_lst)
    acc = sum(l * n for l, n in batch_acc_lst) / sum(n for _, n in batch_acc_lst)
    return loss, acc


def main(_):
    for flag in sorted(FLAGS.__flags):
        tf.logging.info("FLAG OPTION: [{} = {}]".format(flag, str(FLAGS.__flags[flag])))
    train()


if __name__ == "__main__":
    np.random.seed(666)
    tf.set_random_seed(666)
    tf.app.run()

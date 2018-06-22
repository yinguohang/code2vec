# -*- coding:utf-8 -*-

# 主程序入口
import os
import sys
import time

if sys.version[0] == '2':
    from Queue import PriorityQueue
else:
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
        self.node_embedding_size = FLAGS.node_embedding_size
        self.path_embedding_size = FLAGS.path_embedding_size
        self.encode_size = FLAGS.encode_size
        self.dropout_rate = FLAGS.dropout_rate
        self.classification = FLAGS.classification
        self.attention_layer_dimension = FLAGS.attention_layer_dimension
        self.encoding_layer_penalty_rate = FLAGS.encoding_layer_penalty_rate
        self.attention_layer_penalty_rate = FLAGS.attention_layer_penalty_rate
        self.regression_layer_penalty_rate = FLAGS.regression_layer_penalty_rate
        self.node_cnt = reader.node_converter.cnt + 1
        self.path_cnt = reader.path_converter.cnt + 1


def train():
    reader = DataReader(os.path.join(FLAGS.data_path, FLAGS.data_set), FLAGS.context_bag_size)

    train_data = reader.train_dataset
    eval_data = reader.dev_dataset

    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

    batch_data = iterator.get_next()
    start = batch_data['start']
    path = batch_data['path']
    end = batch_data['end']
    score = batch_data['score']

    train_init_op = iterator.make_initializer(train_data)
    eval_init_op = iterator.make_initializer(eval_data)

    with tf.variable_scope("code2vec_model"):
        opt = Option(reader)
        train_model = Code2VecModel(start, path, end, score, opt)
        train_op = utils.get_optimizer(FLAGS.optimizer).minimize(train_model.loss)

    with tf.variable_scope('code2vec_model', reuse=True):
        eval_opt = Option(reader, training=False)
        eval_model = Code2VecModel(start, path, end, score, eval_opt)

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

            tf.logging.info(
                'Epoch %d: train-loss: %.8f (acc=%.2f), val-loss: %.8f (acc=%.2f), min-loss: %.8f, cost-time: %.4f s'
                % (i + 1, train_loss, train_acc, eval_loss, eval_acc, -np.mean(min_eval_loss.queue),
                   time.time() - start_time))

            if stable_min_loss >= 5 and i > 200: break


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
    tf.logging.info(str(FLAGS.__flags))
    train()


if __name__ == "__main__":
    np.random.seed(123)
    tf.set_random_seed(123)
    tf.app.run()

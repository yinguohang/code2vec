# -*-coding:utf-8-*-

import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction


class Code2VecModel:
    def __init__(self, start, path, end, score, opt):
        with tf.device('/cpu:0'):

            self.regularizations = {}

            embedding_outputs = self.build_embedding_layer(start, path, end, opt)

            encoding_outputs = self.build_encoding_layer(embedding_outputs, opt)

            mask = tf.logical_not(tf.equal(start, 0))
            attention_outputs = self.build_attention_layer(encoding_outputs, mask, opt)

            regression_outputs = self.build_regression_layer(attention_outputs, opt)

            if opt.classification > 0:

                self.loss = tf.losses.sigmoid_cross_entropy(
                    self.regression_to_classification(score, opt.classification),
                    tf.reshape(regression_outputs, [-1, opt.classification]),
                    reduction=Reduction.MEAN)
                self.acc = tf.metrics.accuracy(
                    labels=tf.argmax(self.regression_to_classification(score, opt.classification), 1),
                    predictions=tf.argmax(tf.reshape(regression_outputs, [-1, opt.classification]), 1))
            else:
                self.loss = tf.losses.mean_squared_error(score, tf.reshape(regression_outputs, [-1]))
                self.acc = tf.zeros(1)

            if opt.training:
                self.loss += tf.add_n(self.regularizations.values())

    def regression_to_classification(self, inputs, category_cnt):
        return tf.one_hot(tf.cast(tf.floor(inputs * category_cnt), dtype=tf.int32), category_cnt)

    # Embedding
    def build_embedding_layer(self, start, path, end, opt):
        with tf.variable_scope('embedding'):
            node_embedding = tf.get_variable("node_embedding",
                                             [opt.node_cnt, opt.node_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=tf.float32)
            path_embedding = tf.get_variable("path_embedding",
                                             [opt.path_cnt, opt.path_embedding_size],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=tf.float32)

            embedding_start = tf.nn.embedding_lookup(node_embedding, start)
            embedding_path = tf.nn.embedding_lookup(path_embedding, path)
            embedding_end = tf.nn.embedding_lookup(node_embedding, end)

            outputs = tf.concat([embedding_start, embedding_path, embedding_end], axis=2)

            if opt.training:
                tf.logging.info("Building Code2VecModel - {:16s}: ({}, {}, {}) -> ({})"
                                .format("Embedding Layer", start.get_shape(), path.get_shape(),
                                        end.get_shape(), outputs.get_shape()))
            return outputs

    # Encoding
    def build_encoding_layer(self, inputs, opt):
        with tf.variable_scope('encoding'):
            bag_size = inputs.get_shape()[1]
            context_size = inputs.get_shape()[2]

            # FC (1 layer, no activation)
            inputs_flatten = tf.reshape(inputs, [-1, context_size])
            inputs_dropout = tf.layers.dropout(inputs_flatten,
                                               rate=opt.dropout_rate,
                                               training=opt.training)

            encoding_fc_weight = tf.get_variable('encoding_fc_weight',
                                                 [context_size, opt.encode_size],
                                                 initializer=tf.contrib.layers.xavier_initializer(),
                                                 dtype=tf.float32)

            outputs_flatten = tf.nn.tanh(tf.matmul(inputs_dropout, encoding_fc_weight))

            self.regularizations['encoding_fc_weight_L2'] = tf.norm(encoding_fc_weight, ord=2) * 0.3

            outputs = tf.reshape(outputs_flatten, [-1, bag_size, opt.encode_size])

            if opt.training:
                tf.logging.info("Building Code2VecModel - {:16s}: ({}) -> ({})"
                                .format("Encoding Layer", inputs.get_shape(), outputs.get_shape()))
            return outputs

    # Attention layer
    def build_attention_layer(self, inputs, mask, opt):
        with tf.variable_scope("attention"):
            bag_size = inputs.get_shape()[1]
            attention_dimension = 5

            inputs_flatten = tf.reshape(inputs, [-1, opt.encode_size])
            inputs_dropout = tf.layers.dropout(inputs_flatten,
                                               rate=opt.dropout_rate,
                                               training=opt.training)

            attention_weight = tf.get_variable('attention_weight',
                                               [attention_dimension, opt.encode_size],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               dtype=tf.float32)

            attention_value = tf.get_variable('attention_value',
                                              [attention_dimension, 1],
                                              initializer=tf.contrib.layers.xavier_initializer(),
                                              dtype=tf.float32)

            context_weights_flatten = tf.matmul(inputs_dropout, tf.transpose(attention_weight))
            context_weights_flatten = tf.matmul(tf.nn.tanh(context_weights_flatten), attention_value)

            context_weights = tf.reshape(context_weights_flatten, [-1, bag_size, 1])

            context_weights_masked = tf.where(tf.reshape(mask, [-1, bag_size, 1]), context_weights,
                                              tf.multiply(tf.ones_like(context_weights), -3.4e38))

            context_weights_softmax = tf.nn.softmax(context_weights_masked, 1)

            context_weights_repeat = tf.tile(context_weights_softmax, [1, 1, opt.encode_size])

            context_mul = tf.multiply(inputs, context_weights_repeat)

            context_sum = tf.reduce_sum(context_mul, 1)

            # Encourage weights to be orthogonal
            orthogonal_penalty = tf.square(tf.norm(tf.matmul(attention_weight, tf.transpose(attention_weight))
                                                   - tf.eye(attention_dimension), ord='fro', axis=(0, 1)))

            self.regularizations['attention_orthogonal_penalty'] = orthogonal_penalty * 0.1

            if opt.training:
                tf.logging.info("Building Code2VecModel - {:16s}: ({}) -> ({})"
                                .format("Attention Layer", inputs.get_shape(), context_sum.get_shape()))
            return context_sum

    # Regression
    def build_regression_layer(self, inputs, opt):
        with tf.variable_scope("regression"):
            bag_size = inputs.get_shape()[1]
            output_size = opt.classification if opt.classification > 0 else 1

            dropout_inputs_1 = tf.layers.dropout(inputs,
                                                 rate=opt.dropout_rate,
                                                 training=opt.training)

            regression_weight_1 = tf.get_variable("regression_weight_1",
                                                  [bag_size, bag_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  dtype=tf.float32)
            regression_bias_1 = tf.get_variable("regression_bias_1",
                                                [bag_size],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float32)

            outputs_1 = tf.nn.relu(tf.matmul(dropout_inputs_1, regression_weight_1) + regression_bias_1)

            dropout_inputs_2 = tf.layers.dropout(outputs_1,
                                                 rate=opt.dropout_rate,
                                                 training=opt.training)

            regression_weight_2 = tf.get_variable("regression_weight_2",
                                                  [bag_size, output_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  dtype=tf.float32)
            regression_bias_2 = tf.get_variable("regression_bias_2",
                                                [output_size],
                                                initializer=tf.zeros_initializer(),
                                                dtype=tf.float32)

            outputs = tf.matmul(dropout_inputs_2, regression_weight_2) + regression_bias_2

            self.regularizations['regression_L2_1'] = tf.norm(regression_weight_1, ord=2) * 0.002
            self.regularizations['regression_L2_2'] = tf.norm(regression_weight_2, ord=2) * 0.002

            if opt.training:
                tf.logging.info("Building Code2VecModel - {:16s}: ({}) -> ({})"
                                .format("Regression Layer", inputs.get_shape(), outputs.get_shape()))
            return outputs

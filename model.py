# -*-coding:utf-8-*-

import tensorflow as tf


class Code2VecModel:
    def __init__(self, start, path, end, score, opt):
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.node_embedding = tf.get_variable("node_embedding",
                                                  [opt.node_cnt, opt.node_embedding_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  dtype=tf.float32)
            self.path_embedding = tf.get_variable("path_embedding",
                                                  [opt.path_cnt, opt.path_embedding_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  dtype=tf.float32)
            if opt.training:
                tf.summary.histogram('node_embedding', self.node_embedding)
                tf.summary.histogram("path_embedding", self.path_embedding)

            self.mask = tf.logical_not(tf.equal(start, 0))

            embedding_inputs = Code2VecModel.build_input(start, path, end, self.node_embedding, self.path_embedding)
            encoded_inputs = Code2VecModel.build_encode_input(embedding_inputs, opt)
            attention_outputs = Code2VecModel.build_attention(encoded_inputs, self.mask, opt)
            self.loss, self.outputs = Code2VecModel.build_regression(attention_outputs, score, opt)

            if opt.training:
                tf.summary.scalar('loss', self.loss)

    # Embedding
    @staticmethod
    def build_input(start, path, end, node_embedding, path_embedding):
        embedding_start = tf.nn.embedding_lookup(node_embedding, start)
        embedding_path = tf.nn.embedding_lookup(path_embedding, path)
        embedding_end = tf.nn.embedding_lookup(node_embedding, end)
        inputs = tf.concat([embedding_start, embedding_path, embedding_end], axis=2)
        tf.logging.info("Code2VecModel - Embedding output:  %s" % inputs)
        return inputs

    # FC1
    @staticmethod
    def build_encode_input(inputs, opt):
        with tf.name_scope('encode'):
            bag_size = inputs.get_shape()[1]
            context_size = inputs.get_shape()[2]
            encode_weights = tf.get_variable('encode_weights',
                                             [context_size, opt.encode_size],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=tf.float32)
            flatten_input = tf.reshape(inputs, [-1, context_size])
            encode_dropout = tf.layers.dropout(flatten_input, opt.dropout_rate)
            flatten_encode_inputs = tf.nn.tanh(tf.matmul(encode_dropout, encode_weights))
            encode_inputs = tf.reshape(flatten_encode_inputs, [-1, bag_size, opt.encode_size])
            tf.logging.info('Code2VecModel - Encoded output:    %s' % encode_inputs)
            return encode_inputs

    # Attention layer
    @staticmethod
    def build_attention(inputs, mask, opt):
        with tf.name_scope("attention"):
            bag_size = inputs.get_shape()[1]
            encode_size = inputs.get_shape()[2]
            context_weights = tf.contrib.layers.fully_connected(inputs, 1, activation_fn=None)
            context_weights_masked = tf.where(tf.reshape(mask, [-1, bag_size, 1]), context_weights,
                                              tf.multiply(tf.ones_like(context_weights), -3.4e38))
            context_weights_softmax = tf.nn.softmax(context_weights_masked, 1)
            context_weights_repeat = tf.tile(context_weights_softmax, [1, 1, encode_size])
            context_mul = tf.multiply(inputs, context_weights_repeat)
            context_sum = tf.reduce_sum(context_mul, 1)
            tf.logging.info("Code2VecModel - Attention output:  %s" % context_sum)
            return context_sum

    # Regression
    @staticmethod
    def build_regression(inputs, score, opt):
        with tf.name_scope("regression"):
            batch_size = inputs.get_shape()[0]
            bag_size = inputs.get_shape()[1]
            output_weights = tf.get_variable("output_weights",
                                             [bag_size, 1],
                                             initializer=tf.contrib.layers.xavier_initializer(),
                                             dtype=tf.float32)
            dropout_inputs = tf.layers.dropout(inputs,
                                               rate=opt.dropout_rate,
                                               training=opt.training)
            outputs = tf.matmul(dropout_inputs, output_weights)
            loss = tf.losses.mean_squared_error(score, tf.reshape(outputs, [-1]))
            tf.logging.info("Code2VecModel - Regression output: %s" % outputs)
            return loss, outputs

# -*- coding:utf-8 -*-
import pickle

import numpy as np
import tensorflow as tf


class Converter:
    def __init__(self):
        self.cnt = 0
        self.names = []
        self.name2index = {}

    def load(self, name):
        with tf.gfile.GFile(name, "r") as f:
            for line in f.readlines():
                line = line.strip()
                temp = line.split(",")
                self.names.append(temp[1])
                self.name2index[temp[1]] = int(temp[0])
            self.cnt = len(self.names)

    def get_index(self, s):
        if s in self.names:
            return self.name2index[s]
        self.names.append(s)
        # 从1开始
        self.name2index[s] = self.cnt + 1
        self.cnt = self.cnt + 1
        return self.cnt

    def get_name(self, i):
        if i > self.cnt:
            return None
        else:
            return self.names[i - 1]


class DataReader:
    def __init__(self, input_file_name, context_bag_size):
        if tf.gfile.Exists(input_file_name + "-" + str(context_bag_size) + "-data-X.npy"):
            load = True
        else:
            load = False
        self.node_converter = Converter()
        self.node_converter.load(input_file_name + "-node.txt")
        self.path_converter = Converter()
        self.path_converter.load(input_file_name + "-path.txt")
        if not load:
            tf.logging.info("Read from original files...")
            self.data_X = None
            self.data_y = None
            self.m = 0
            # self.node_converter = Converter()
            # self.node_converter.load(input_file_name + "-node.txt")
            # self.path_converter = Converter()
            # self.path_converter.load(input_file_name + "-path.txt")
            self.read_corpus(input_file_name, context_bag_size)
            with tf.gfile.Open(input_file_name + "-" + str(context_bag_size) + "-data-X.npy", "wb") as f:
                np.save(f, self.data_X)
            with tf.gfile.Open(input_file_name + "-" + str(context_bag_size) + "-data-y.npy", "wb") as f:
                np.save(f, self.data_y)
            # with tf.gfile.Open(input_file_name + "-node-converter.pkl", "wb") as f:
            #     pickle.dump(self.node_converter, f)
            # with tf.gfile.Open(input_file_name + "-path-converter.pkl", "wb") as f:
            #     pickle.dump(self.path_converter, f)
        else:
            tf.logging.info("Read from cache...")
            with tf.gfile.Open(input_file_name + "-" + str(context_bag_size) + "-data-X.npy", "rb") as f:
                self.data_X = np.load(f)
            with tf.gfile.Open(input_file_name + "-" + str(context_bag_size) + "-data-y.npy", "rb") as f:
                self.data_y = np.load(f)
            # with tf.gfile.Open(input_file_name + "-node-converter.pkl", "rb") as f:
            #     self.node_converter = pickle.load(f)
            # with tf.gfile.Open(input_file_name + "-path-converter.pkl", "rb") as f:
            #     self.path_converter = pickle.load(f)
            self.m = self.data_X.shape[0]
        self.generate_dataset()

    def read_corpus(self, input_file_name, context_bag_size):
        f = tf.gfile.Open(input_file_name + "-context.txt")
        data_X = []
        data_y = []
        original_features = []
        X = np.zeros((0, 3))
        lines = f.readlines()
        total = len(lines)
        for i in range(0, total):
            # 删除掉两端的空白字符
            line = lines[i].strip()
            if int(i / total * 100) > int((i - 1) / total * 100):
                tf.logging.info("%d percent" % int(i / total * 100))
            if line.startswith("method_name") or line.startswith("class_name"):
                # 如果上一个X为空，则直接丢弃
                if X.shape[0] == 0:
                    if len(data_y) > 0:
                        data_y.pop()
                        original_score.pop()
                    continue
                # 如果长度不够，则使用0来padding
                if X.shape[0] < context_bag_size:
                    X = np.r_[X, np.zeros((context_bag_size - X.shape[0], 3))]
                data_X.append(X)
                X = np.zeros((0, 3))
                pass
            elif line.startswith("score:"):
                original_score = float(line.split(":")[1])
                if original_score == 0:
                    score = 1.0
                else:
                    score = np.tanh(4 / original_score)
                data_y.append(score)
            elif line.startswith("features:"):
                features = list(filter(lambda x: x != None, map(lambda x: None if x == "None" else float(x), line.split(":")[1].split(","))))
                original_features.appnd(features)
            else:
                if X.shape[0] >= context_bag_size:
                    continue
                temp = line.split(",")
                # start = self.node_converter.get_index(temp[0])
                # path = self.path_converter.get_index(temp[1])
                # end = self.node_converter.get_index(temp[2])
                start = int(temp[0])
                path = int(temp[1])
                end = int(temp[2])
                X = np.r_[X, np.array([[start, path, end]])]
        tf.logging.info("Loaded %d lines." % total)
        if X.shape[0] < context_bag_size:
            X = np.r_[X, np.zeros((context_bag_size - X.shape[0], 3))]
        data_X.append(X)
        self.data_X = np.array(data_X)
        self.data_y = np.array(data_y)
        self.original_features = original_features
        self.m = self.data_y.shape[0]
        print(self.data_X.shape)
        print(self.data_y.shape)
        random_index = np.random.permutation(self.m)
        self.data_X = self.data_X[random_index]
        self.data_y = self.data_y[random_index]

    @staticmethod
    def np2tf(X, y):
        dataset = {}
        dataset["start"] = X[:, :, 0].astype(np.int32)
        dataset["path"] = X[:, :, 1].astype(np.int32)
        dataset["end"] = X[:, :, 2].astype(np.int32)
        dataset["score"] = y
        return tf.data.Dataset.from_tensor_slices(dataset)

    def generate_dataset(self):
        self.train_X, self.train_y = self.data_X[:int(self.m * 0.6), :, :], self.data_y[:int(self.m * 0.6)]
        self.dev_X, self.dev_y = self.data_X[int(self.m * 0.6):int(self.m * 0.8), :, :], self.data_y[
                                                                                         int(self.m * 0.6):int(
                                                                                             self.m * 0.8)]
        self.test_X, self.test_y = self.data_X[int(self.m * 0.8):, :, :], self.data_y[int(self.m * 0.8):]
        tf.logging.info("Training set: %s" % str(self.train_X.shape))
        tf.logging.info("Validation set: %s" % str(self.dev_X.shape))
        tf.logging.info("Test set: %s" % str(self.test_X.shape))
        self.train_dataset = DataReader.np2tf(self.train_X, self.train_y).shuffle(buffer_size=60).batch(32)
        self.dev_dataset = DataReader.np2tf(self.dev_X, self.dev_y).shuffle(buffer_size=60).batch(32)
        self.test_dataset = DataReader.np2tf(self.test_X, self.test_y).shuffle(buffer_size=60).batch(32)


if __name__ == "__main__":
    reader = DataReader("temp.txt", 100)
    tf.logging.info(reader.train_dataset)

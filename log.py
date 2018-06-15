#-*- coding:utf-8 -*-

import tensorflow as tf
import time

class Logger:
    @classmethod
    def set_file(cls, prefix):
        cls.f = tf.gfile.Open(prefix + "-log-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".txt", "w")

    @classmethod
    def print(cls, s):
        print(s)
        cls.f.write("%s\n" % s)
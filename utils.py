import tensorflow as tf

def get_optimizer(name):
    if name == "adadelta":
        return tf.train.AdadeltaOptimizer(1.0)
    if name == "adam":
        return tf.train.AdamOptimizer(0.001)
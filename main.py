import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import models.code2vec.trainer as code2vec_trainer
from utils import config, utils

FLAGS = tf.app.flags.FLAGS

config.init()
utils.init_tf_logging(FLAGS.log_path)

for flag in sorted(FLAGS.__flags):
    tf.logging.info("FLAG OPTION: [{} = {}]".format(flag, str(FLAGS.__flags[flag])))

code2vec_trainer.train()

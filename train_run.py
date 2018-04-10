from model.run_epoch import *
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('ckpt_path', 'purpose_ckpt', '模型保存路径')
flags.DEFINE_string('corpus_path', 'yy_ckpt', '训练语料路径')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':
    tf.app.run(run_epoch(FLAGS))


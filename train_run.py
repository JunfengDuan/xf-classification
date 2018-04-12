from model.run_epoch import *
import tensorflow as tf
from simple_demo import run_epoch1

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string('ckpt_path', 'yy_ckpt', '模型保存路径')
flags.DEFINE_string('corpus_path', os.path.join('datafile', 'xf_yy_data'), '训练语料路径')
flags.DEFINE_string('model_name', 'yy', '模型名称')
flags.DEFINE_string('tensorboard_dir', os.path.join('tensorboard', FLAGS.model_name), 'tensorboard路径')


if __name__ == '__main__':
    tf.app.run(run_epoch(FLAGS))


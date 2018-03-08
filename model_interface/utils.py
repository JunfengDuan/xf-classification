import tensorflow as tf
import logging
import os
import shutil
import tensorflow.contrib.keras as kr
from loader.xf_data_load import read_vocab, read_category
import numpy as np


def save_model(sess, model, path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    checkpoint_path = os.path.join(path, "model.ckpt")
    save_path = model.saver.save(sess, checkpoint_path)

    return save_path


def restore_model(session, model_class, config, path, logger):

    model = model_class(config)

    ckpt = tf.train.get_checkpoint_state(path)
    save_path = ckpt.model_checkpoint_path
    if ckpt and tf.train.checkpoint_exists(save_path):
        logger.info("Reading model parameters from %s" % save_path)
        model.saver.restore(session, save_path)
    else:
        logger.info("Can't find the checkpoint, stopping...")

    return model


def evaluate_text(sess, model, input_text):

    words, word_to_id = read_vocab('datafile/vocab/xf_vocab.txt')
    _, cat_to_id = read_category()

    text_id = []
    for word in list(input_text):
        if word in word_to_id:
            text_id.append(word_to_id[word])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences([text_id], 600)
    # 将标签转换为one-hot表示
    # y_pad = kr.utils.to_categorical(list(cat_to_id.values()))
    index2label = dict(zip(cat_to_id.values(), cat_to_id.keys()))

    logits = sess.run(model.logits, feed_dict={model.input_x: x_pad})
    predicted_labels, value_labels = decode(logits[0], index2label)
    value_labels_exp = np.exp(value_labels)
    p_labels = value_labels_exp / np.sum(value_labels_exp)

    return predicted_labels[0], p_labels[0]


def decode(logits, index2label, top_number=5):
    """
    get label using logits with value
    :param logits:
    :param index2label:
    :param top_number:
    :return:
    """
    index_list = np.argsort(logits)[-top_number:]  # print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list = index_list[::-1]
    value_list = []
    label_list = []
    for index in index_list:
        label = index2label[index]
        label_list.append(label)  # ('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
        value_list.append(logits[index])
    return label_list, value_list


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def write_to_file(filename, text_list):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(root_dir, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
    os.mknod(file_path)

    with open(filename, 'w', encoding='utf-8') as f:
        for text in text_list:
            f.write(text + '\n')

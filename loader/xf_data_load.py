#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow.contrib.keras as kr
import numpy as np
import os
from data_preprocess.build_vocab import read_file


def read_vocab(filename):
    """
    读取词汇表
    words=['a', 'b']
    word_to_id=['a':'0','b':'1']
    :param filename:
    :return:
    """
    words = list(map(lambda line: line.strip(),
                     open(filename, 'r', encoding='utf-8').readlines()))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id


def read_category():
    """
    读取分类目录，固定
    :return:
    """
    # categories = ['申诉', '求决', '意见建议', '揭发控告', '其他']
    categories = ['是', '否']

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """
    将id表示的内容转换为文字
    :param content:
    :param words:
    :return:
    """
    return ''.join(words[x] for x in content)


def file_to_ids(contents, labels, word_to_id, max_length=600):
    """
    将文件转换为id表示
    data_id = [[1,0,3,2,5], [2,3,4,3,7,8,8], [5,6,7]]
    label_id = [1,2,3]
    x_pad = [10325   ,2343788 ,567     ]
    y_pad = [001,010,100]
    :param contents:
    :param labels:
    :param word_to_id:
    :param max_length:
    :return:
    """
    _, cat_to_id = read_category()

    data_id = []
    label_id = []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    # 将标签转换为one-hot表示
    y_pad = kr.utils.to_categorical(label_id)

    return x_pad, y_pad


def process_file(data_path='datafile', corpus_path='corpus', seq_length=600):
    """
    一次性返回所有数据
    :param corpus_path:
    :param data_path:
    :param seq_length:
    :return:
    """

    words, word_to_id = read_vocab(os.path.join(data_path, 'vocab/xf_vocab.txt'))

    corpus_path = os.path.join(data_path, corpus_path)
    train_contents, train_labels = read_file(os.path.join(corpus_path, 'xf_train.txt'))
    test_contents, test_labels = read_file(os.path.join(corpus_path, 'xf_test.txt'))
    val_contents, val_labels = read_file(os.path.join(corpus_path, 'xf_val.txt'))

    x_train, y_train = file_to_ids(train_contents, train_labels, word_to_id, seq_length)
    x_test, y_test = file_to_ids(test_contents, test_labels, word_to_id, seq_length)
    x_val, y_val = file_to_ids(val_contents, val_labels, word_to_id, seq_length)

    return x_train, y_train, x_test, y_test, x_val, y_val, words


def batch_iter(data, batch_size=64, num_epochs=5):
    """
    生成批次数据
    :param data:
    :param batch_size:
    :param num_epochs:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batch_per_epoch = int((data_size - 1) / batch_size) + 1  # 批数
    for epoch in range(num_epochs):  # 迭代num_epochs次
        indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[indices]  # 如果一个批次里样本的相关性太大，会导致模型的泛化能力不好，需要洗牌

        for batch_num in range(num_batch_per_epoch):
            # print("batch_num :", batch_num)
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

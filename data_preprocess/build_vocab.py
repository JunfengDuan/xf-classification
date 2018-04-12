#!/usr/bin/python
# -*- coding: utf-8 -*-

import codecs
from collections import Counter
import os


def _create_vocab_file(vocab_name):
    """
    创建词汇表目录及空文件
    :param vocab_name:
    :return:
    """
    root_dir = os.path.dirname(os.path.dirname(__file__))
    vocab_parent_path = os.path.join(root_dir, 'datafile/vocab')
    vocab_path = os.path.join(vocab_parent_path, vocab_name)
    if not os.path.exists(vocab_parent_path):
        os.makedirs(vocab_parent_path)
    if os.path.exists(vocab_path):
        os.remove(vocab_path)
    os.mknod(vocab_path)
    return vocab_path


def read_file(filename):
    """
    读取文件数据
    :param filename:
    :return:contents=[[我,家,住], [], ...] labels=[神书,球觉]
    """
    contents = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                label, content = line.strip().split('\t')
                contents.append(list(content))
                labels.append(label)
            except:
                pass
    return contents, labels


def _build_vocab(train_file, vocab_file, vocab_size=5000):
    """
    构建词汇表
    :param train_file: 训练集
    :param vocab_file: 构造向量所用的词汇表
    :param vocab_size:
    :return:
    """
    print("xf_file:", train_file)
    print("vocab_file:", vocab_file)
    """根据训练集构建词汇表，存储"""
    data, _ = read_file(train_file)

    all_data = []
    for content in data:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    codecs.open(vocab_file, 'w', 'utf-8').write('\n'.join(words))
    print("vocab_file is build, words of vocab is:", len(words))


if __name__ == '__main__':

    vocab_file_path = _create_vocab_file('xf_vocab.txt')
    project_root_dir = os.path.dirname(os.path.dirname(__file__))
    train_data_path = os.path.join(project_root_dir, 'datafile/xf_purpose_data/xf_train.txt')
    _build_vocab(train_data_path, vocab_file_path)


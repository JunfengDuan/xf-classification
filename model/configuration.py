#!/usr/bin/python
# -*- coding: utf-8 -*-


class TCNNConfig(object):
    """CNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 5        # 类别数

    num_filters = 128       # 卷积核数目
    kernel_size = 5         # 卷积核尺寸
    vocab_size = 5000       # 词汇表达小

    hidden_dim = 128        # 全连接层神经元
    dropout_keep_prob = 0.8  # dropout保留比例 默认0.5效果就很好
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次


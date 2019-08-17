#!/usr/bin/env python
# coding: utf-8
"""
@File     :data_helper.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/11
@Desc     : 
"""
import itertools
import numpy as np
from collections import Counter
from tools.dataio import load_txt_data
from pre_process_data import build_data_set


def build_seq_data_set(seq_len):
    raw_data = load_txt_data('./data/raw_data.csv')
    raw_data += load_txt_data('./data/new_raw_data.txt')
    x_train = []
    y_train = []
    ptr = 0

    while ptr + seq_len < len(raw_data):
        delta = 0
        tmp_x = []
        # tmp_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tmp_y = [0, 0]  # binary classification
        while len(tmp_x) < seq_len:
            tmp_x.append(raw_data[ptr + delta])
            delta += 1
        # print('delta:', delta)
        # print('ptr:', ptr)
        # print('slice:', tmp_x)
        # print('compare slice:', raw_data[ptr:ptr+seq_len+1])
        # print('label:', raw_data[ptr+seq_len])
        x_train.append(tmp_x)

        if 3 >= int(raw_data[ptr + seq_len]) > 0:
            tmp_y[0] = 1
        else:
            tmp_y[1] = 1
        # tmp_y[int(raw_data[ptr + seq_len])] = 1  # 10 classification
        # print(tmp_y)
        y_train.append(tmp_y)
        # ptr += (delta+1)
        ptr += 1

    # print(len(x_train), len(y_train))
    return x_train, y_train


def build_seq_data_set2():
    raw_data = load_txt_data('./data/format_data.txt')
    print(len(raw_data))
    stack = []
    x = []
    y = []

    for raw in raw_data:
        if not raw:
            x.append(stack)
            stack = []
        else:
            raw = raw.split(' ')
            data = raw[0]

            # stack.append(train)

    return


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)

    return [x, y]


def load_data(seq_len):
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = build_seq_data_set(seq_len)
    # sentences, labels = build_seq_data_set2()

    vocabulary, vocabulary_inv = build_vocab(sentences)
    x, y = build_input_data(sentences, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


if __name__ == '__main__':
    _, c, v, vi = load_data(5)
    print(_)
    # print(build_seq_data_set2(5, 10))

#!/usr/bin/env python
# coding: utf-8
"""
@File     :data_helper.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/16
@Desc     :
"""
import os
import pickle
import numpy as np
from collections import Counter
import platform
from keras.preprocessing.sequence import pad_sequences


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data


def load_data(dic_config):
    train = _parse_data(open('./data/format_data.txt', 'rb'))
    # print(train)

    word_counts = Counter(row[0].lower() for sample in train for row in sample)
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    # print(vocab)
    print(len(vocab))
    # chunk_tags = ['O_0', 'O_1', 'O_2', 'O_3', 'O_4',
    #               'O_5', 'O_6', 'O_7', 'O_8', 'O_9']
    chunk_tags = ['O_0', 'O_1']
    # save initial config data
    with open(dic_config, 'wb') as outp:
        pickle.dump((vocab, chunk_tags), outp)

    train = _process_data(train, vocab, chunk_tags)
    return train, (vocab, chunk_tags)


def _process_data(data, vocab, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
        print(maxlen)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0].lower(), 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    # print(y_chunk)

    # for s in data:
    #     for w in s:
    #         try:
    #             a = chunk_tags.index(w[1])
    #         except IndexError:
    #             print(s)
    #             print(w)

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = np.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, vocab, maxlen=288):
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [word2idx.get(w[0].lower(), 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length

#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: predict.py
@time: 8/29/2019 9:51 AM
@desc:
"""
import numpy
import keras
import pickle
from collections import OrderedDict


def word2id(word):
    with open('./model/vocab_config.pkl', 'rb') as inp:
        vocabulary = pickle.load(inp)

    id_list = []
    for feat in word:
        try:
            id_list.append(vocabulary[str(feat)])
        except KeyError:
            id_list.append(vocabulary['unk'])

    return numpy.array([id_list])


def predict_one():
    pass


def predict_ui(last_sequence):
    _model = keras.models.load_model('./model/res.model')

    while True:

        _embed = word2id(last_sequence)

        y_predict = _model.predict(_embed)

        _id = []
        _prob = []
        for i, x in enumerate(list(y_predict[0])):
            _id.append(i)
            _prob.append(x)

        res_dict = dict(zip(_id, _prob))
        res_dict = sorted(res_dict.items(), key=lambda item: item[1], reverse=True)

        for pair in res_dict:
            pair = list(pair)

            print("预测值: {}, 概率: {}".format(pair[0], pair[1]))

        print('*' * 88)
        print('请输入最新的真实值, 格式: 时间点+星期几+值, (28879)')
        usr_input = input()
        last_sequence.append(usr_input)
        last_sequence.pop(0)


if __name__ == '__main__':
    _last_sequence = [27925, 28021, 28126, 28228, 28323, 28425, 28529, 28629, 28726, 28829]

    predict_ui(_last_sequence)

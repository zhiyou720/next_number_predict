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
from tools.dataio import load_txt_data, save_txt_file
from config import train_data_path, MODEL_PATH


class ResultAnalysis:
    def __init__(self, x_open_test, y_open_test, _max_predict, _future, last_sequence=None):
        print('Test...')
        self.model = keras.models.load_model('./model/res.model')

        self.last_sequence = last_sequence

        self.x_open_test = x_open_test
        self.y_open_test = y_open_test

        self.max_predict = _max_predict
        self.future = _future

    def predict_analyze(self):
        if not self.last_sequence:
            y_predict_prob_matrix = self.model.predict(self.x_open_test)
            y_predict_ordered_by_prob = self.sort_index_by_value(y_predict_prob_matrix)
            y_true = self.get_value_from_softmax_matrix(self.y_open_test)

            self.predict_from_one_to_max_predict(y_true, y_predict_ordered_by_prob)
            self.comprehensive_predict(y_true, y_predict_ordered_by_prob)
        else:
            raise NotImplementedError

    def predict_one(self, sequence):
        y_predict = self.model.predict(sequence)

    def comprehensive_predict(self, y_true, y_predict):
        score = 0
        for i in range(len(y_true)):
            if y_true[i] == y_predict[i][0]:
                score += 1
                print('真实值: {}, 预测值: {}, 精准预测成功'.format(y_true[i], y_predict[i][:self.max_predict]))
            elif y_true[i] in y_predict[i][:self.max_predict]:
                score += 1
                print('真实值: {}, 预测值: {}, 预测成功'.format(y_true[i], y_predict[i][:self.max_predict]))
            else:
                ptr = 0
                while ptr <= self.max_predict:
                    if ptr == self.max_predict:
                        print('真实值: {}, 预测值: {}, 预测失败'.format(y_true[i],
                                                              y_predict[i][:self.max_predict]))
                        break
                    elif y_predict[i][ptr] in y_true[i:i + self.future]:
                        score += 1
                        print('真实值: {}, 预测值: {}, 其中 {} 将在未来 {} 行之内出现'
                              .format(y_true[i], y_predict[i][:self.max_predict],
                                      y_predict[i][ptr], self.future))
                        break
                    else:
                        ptr += 1
        print('预测概率 {}'.format(score / len(self.x_open_test)))

    def predict_from_one_to_max_predict(self, y_true, res_index):
        for rng in range(self.max_predict):
            score = 0
            for i in range(len(y_true)):
                if y_true[i] in res_index[i][:rng + 1]:
                    score += 1
            print('预测概率前 {} 个数: {}'.format(rng + 1, score / len(self.y_open_test)))

    @staticmethod
    def bubble_sort(arr):
        n = len(arr)
        for _i in range(n):
            for j in range(0, n - _i - 1):
                if arr[j] < arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

    def sort_index_by_value(self, tmp_list):
        res = []
        for item in tmp_list:
            item = list(item)
            tmp = item[:]
            self.bubble_sort(tmp)
            __ = []
            for i in range(len(tmp)):
                __.append(item.index(tmp[i]))
            res.append(__)
        return res

    @staticmethod
    def get_value_from_softmax_matrix(matrix):
        y_true = []
        for _y in matrix:
            _ = list(_y).index(1)
            y_true.append(_)
        return y_true


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


def predict_ui(_last_sequence, _future_sequence):
    _model = keras.models.load_model(MODEL_PATH)

    res = []
    for feature in _future_sequence:

        _embed = word2id(_last_sequence)

        y_predict = _model.predict(_embed)

        _id = []
        _prob = []
        for i, x in enumerate(list(y_predict[0])):
            _id.append(i)
            _prob.append(x)

        res_dict = dict(zip(_id, _prob))
        res_dict = sorted(res_dict.items(), key=lambda item: item[1], reverse=True)

        stack = []

        for pair in res_dict:
            pair = list(pair)

            stack.append("预测: {}, 概率: {}".format(pair[0], pair[1]))
        stack.insert(0, '真实值: {}'.format(feature[-1]))
        res.append(','.join(stack))
        print('真实值: {},'.format(feature[-1]), '{}'.format(stack[1]))

        _last_sequence.append(feature)
        _last_sequence.pop(0)

    return res


if __name__ == '__main__':
    raw_future_sequence = load_txt_data('./data/future_data_a.txt')
    future_sequence = []
    for item in raw_future_sequence:
        raw = item.split(',')
        future_sequence.append(raw[0] + raw[1])

    raw_last_sequence = load_txt_data(train_data_path)

    train_sequence = []
    for item in raw_last_sequence:
        raw = item.split(',')
        train_sequence.append(raw[0] + raw[1])
    last_sequence = train_sequence[-10:]
    r = predict_ui(last_sequence, future_sequence)

    save_txt_file(r, './res/future_predict.csv', encoding='gbk')

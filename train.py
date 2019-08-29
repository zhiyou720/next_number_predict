#!/usr/bin/env python
# coding: utf-8
"""
@File     :train.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/11
@Desc     : 
"""
import keras
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split

from data_helper import DataLoader
from model import TextAttBiRNN


class Predict:
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
            pass

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


if __name__ == '__main__':
    train = False
    GPU = False
    USE_ALL_DATA = False

    batch_size = 32
    embedding_dims = 200
    epochs = 2

    max_len = 10
    class_num = 10
    train_data_path = './data/new_train_diff+week.csv'
    print('Loading data...')

    if train:
        data = DataLoader(train_data_path, train=True, seq_len=max_len, class_num=class_num)
    else:
        data = DataLoader(train_data_path, train=False, seq_len=max_len, class_num=class_num)
    x, y, vocab = data.x_train, data.y_train, data.vocabulary

    if USE_ALL_DATA:
        cx = x
        cy = y
    else:
        cx = x[:-288]
        cy = y[:-288]

    ox = x[-288:]
    oy = y[-288:]

    x_train, x_test, y_train, y_test = train_test_split(cx, cy, test_size=0.01, shuffle=False, random_state=123)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Build model...')
    model = TextAttBiRNN(max_len, len(vocab), embedding_dims, class_num=class_num, GPU=GPU).get_model()
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    if train:
        print('Train...')
        model.fit(cx, cy,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=False,
                  validation_data=(x_test, y_test)
                  )

        model.save('./model/res.model')
    else:

        last_sequence = cx[-1]
        max_predict = 3
        future = 5
        s = Predict(ox, oy, max_predict, future)
        s.predict_analyze()

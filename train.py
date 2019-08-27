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
from model import TextAttBiRNN
from data_helper import DataLoader
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split


def predict(x_open_test, y_open_test, _max_predict, _future):
    _model = keras.models.load_model('./model/res.model')

    score = _model.evaluate(x_open_test, y_open_test, batch_size=16)
    print(score)
    print('Test...')

    result = _model.predict(x_open_test)

    def bubble_sort(arr):
        n = len(arr)
        for _i in range(n):
            for j in range(0, n - _i - 1):

                if arr[j] < arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

    res = []
    for item in result:
        item = list(item)
        tmp = item[:]
        bubble_sort(tmp)
        __ = []
        for i in range(len(tmp)):
            __.append(item.index(tmp[i]))
        res.append(__)

    y_true = []

    for _y in y_open_test:
        _ = list(_y).index(1)
        y_true.append(_)

    total = len(x_open_test)

    for rng in range(5):
        score = 0
        for i in range(len(y_true)):
            if y_true[i] in res[i][:rng + 1]:
                score += 1
        print('预测概率前 {} 个数: {}'.format(rng + 1, score / total))

    score = 0
    for i in range(len(y_true)):

        if y_true[i] == res[i][0]:
            score += 1
            print('真实值: {}, 预测值: {}, 精准预测成功'.format(y_true[i], res[i][:_max_predict]))
        elif y_true[i] in res[i][:_max_predict]:
            score += 1
            print('真实值: {}, 预测值: {}, 预测成功'.format(y_true[i], res[i][:_max_predict]))
        else:
            ptr = 0
            while ptr <= _max_predict:
                if ptr == _max_predict:
                    print('真实值: {}, 预测值: {}, 预测失败'.format(y_true[i], res[i][:_max_predict]))
                    break
                elif res[i][ptr] in y_true[i:i + _future]:
                    score += 1
                    print('真实值: {}, 预测值: {}, 其中 {} 将在未来 {} 行之内出现'
                          .format(y_true[i], res[i][:_max_predict], res[i][ptr], _future))
                    break
                else:
                    ptr += 1
    print('预测概率 {}'.format(score / total))

    return score / total


if __name__ == '__main__':
    train = False
    GPU = True
    batch_size = 32
    embedding_dims = 200
    epochs = 12

    max_len = 10
    class_num = 10
    train_data_path = './data/new_train_diff+week.csv'
    print('Loading data...')

    if train:
        data = DataLoader(train_data_path, train=True, seq_len=max_len, class_num=class_num)
    else:
        data = DataLoader(train_data_path, train=False, seq_len=max_len, class_num=class_num)
    x, y, vocab = data.x_train, data.y_train, data.vocabulary

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
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=False,
                  validation_data=(x_test, y_test))

        model.save('./model/res.model')
    else:
        max_predict = 3
        future = 5
        s = predict(ox, oy, max_predict, future)

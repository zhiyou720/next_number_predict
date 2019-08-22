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
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.losses import categorical_crossentropy
from data_helper import load_data
from model import TextAttBiRNN
import pickle
from tools.dataio import save_txt_file


def predict():
    model = keras.models.load_model('res.model')
    # model.load_weights('./config.pkl')

    score = model.evaluate(x_test, y_test, batch_size=16)
    print(score)
    print('Test...')
    # x_test = x_test[:10]
    # y_test = y_test[:10]
    result = model.predict(x_test)
    print(result)

    # for item in x_test:
    #     result = model.predict([item])
    #     print(result)
    #     print()

    # print(x_test)
    def bubbleSort(arr):
        n = len(arr)

        # 遍历所有数组元素
        for i in range(n):

            # Last i elements are already in place
            for j in range(0, n - i - 1):

                if arr[j] < arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

    res = []
    for item in result:
        item = list(item)
        tmp = item[:]
        bubbleSort(tmp)
        __ = []
        __.append(item.index(tmp[0]))
        __.append(item.index(tmp[1]))
        __.append(item.index(tmp[2]))
        res.append(__)

    # print(res)

    y_true = []

    for y in y_test:
        _ = list(y).index(1)
        y_true.append(_)
        # print(_)

    score = 0
    total = len(x_test)

    for i in range(len(y_true)):
        # print(res[i], y_true[i])

        if y_true[i] in res[i]:
            score += 1
    print(score / total)

    return score / total


batch_size = 16
embedding_dims = 50
epochs = 100

if __name__ == '__main__':
    maxlen = 10

    print('Loading data...')
    x, y, vocab, vocab_index = load_data(maxlen)
    with open('./config.pkl', 'wb') as out_p:
        pickle.dump(vocab, out_p)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    # print('Pad sequences (samples x time)...')
    #
    # x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=None, dtype='int32', padding='pre',
    #                                                      truncating='pre', value=0.0)
    # x_test = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=None, dtype='int32', padding='pre',
    #                                                     truncating='pre', value=0.0)
    # print('x_train shape:', x_train.shape)
    # print('x_test shape:', x_test.shape)

    print('Build model...')
    model = TextAttBiRNN(maxlen, len(vocab), embedding_dims).get_model()
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    print('Train...')
    early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
    model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              # verbose=0,
              callbacks=[early_stopping],
              # shuffle=True,
              validation_data=(x_test, y_test))

    model.save('res.model')

    s = predict()
    #
    # txt = ['MaxLen: {}\tScore: {}'.format(maxlen, s)]
    #
    # save_txt_file(txt, './exp_res.txt')

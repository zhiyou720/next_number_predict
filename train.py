#!/usr/bin/env python
# coding: utf-8
"""
@File     :train.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/11
@Desc     : 
"""
from model import TextAttBiRNN
from predict import ResultAnalysis
from keras.losses import categorical_crossentropy
from config import max_len, vocab, embedding_dims, class_num, GPU, train, train_set_x, train_set_y, batch_size, \
    epochs, x_valid, y_valid, test_set_x, test_set_y, MODEL_PATH, SHUFFLE

if __name__ == '__main__':
    print('Build model...')
    model = TextAttBiRNN(max_len, len(vocab), embedding_dims, class_num=class_num, GPU=GPU).get_model()
    model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    if train:
        print('Train...')
        model.fit(train_set_x, train_set_y,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=SHUFFLE,
                  validation_data=(x_valid, y_valid)
                  )

        model.save(MODEL_PATH)
    else:

        last_sequence = train_set_x[-1]
        max_predict = 5
        future = 1
        s = ResultAnalysis(test_set_x, test_set_y, max_predict, future)
        s.predict_analyze()

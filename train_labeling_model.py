#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: train.py.py
@time: 6/5/2019 5:22 PM
@desc:
"""
import os
import labeling_model
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed

seed(123)
set_random_seed(2)

exp_number = 1
EMBED_DIM = 200
N_LSTM = 1
HIDDEN_UNITS = 20
LSTM_L2 = 0.
LSTM_DROPOUT = 0.

N_DENSE = 1
DENSE_UNITS = 64
DENSE_L2 = 0.
DENSE_DROPOUT = 0.

BN = None

BATCH_SIZE = 8
EPOCHS = 10

CONFIG_PATH = 'model/config_{}.pkl'.format(exp_number)
MODEL_PATH = 'model/slot_model_{}.h5'.format(exp_number)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    '''train model'''
    print(exp_number)
    model, (train_x, train_y) = labeling_model.create_model(CONFIG_PATH,
                                                            embed_size=EMBED_DIM,
                                                            n_lstm=N_LSTM,
                                                            _hidden_units=HIDDEN_UNITS,
                                                            l2_lstm=LSTM_L2,
                                                            dropout_rate_lstm=LSTM_DROPOUT,

                                                            n_dense=N_DENSE,
                                                            dense_units=DENSE_UNITS,
                                                            l2_dense=DENSE_L2,
                                                            dropout_rate_dense=DENSE_DROPOUT,

                                                            bn=BN,
                                                            train=True)
    model.summary()

    # early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')

    history = model.fit(train_x,
                        train_y,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=0.01,
                        shuffle=False,
                        # callbacks=[EarlyStopping]
                        )
    model.save(MODEL_PATH)


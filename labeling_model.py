#!/usr/bin/env python
# encoding: utf-8
"""
@author: zk
@contact: kun.zhang@nuance.com
@file: bi_lstm_crf.py
@time: 6/4/2019 3:29 PM
@desc:
"""
import keras
import pickle
# from keras.losses import categorical_crossentropy
from labeling_data_loader import load_data
from keras_contrib.layers.crf import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras.engine.base_layer import Layer
from keras import initializers, constraints, regularizers


def create_model(dic_config,
                 embed_size=200,

                 n_lstm=1,
                 _hidden_units=10,
                 l2_lstm=0.,
                 dropout_rate_lstm=0.,

                 n_dense=0,
                 dense_units=64,
                 l2_dense=0.,
                 dropout_rate_dense=0.,

                 bn=True,
                 train=True):
    if train:
        (train_x, train_y), (vocab, chunk_tags) = load_data(dic_config)
    else:
        with open(dic_config, 'rb') as inp:
            (vocab, chunk_tags) = pickle.load(inp)

    model = keras.Sequential()

    '''EMBEDDING'''
    embed = keras.layers.Embedding(len(vocab) + 1,
                                   embed_size,
                                   # mask_zero=True,
                                   trainable=True,
                                   name='embedding')
    model.add(embed)

    # for i in range(n_lstm):
    #     # '''Batch Normalization'''
    #     # if bn:
    #     #     bn = keras.layers.BatchNormalization()
    #     #     model.add(bn)
    #
    #     '''LSTM'''
    #     bi_lstm_layer = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(
    #         _hidden_units,
    #         # dropout=dropout_rate_lstm,
    #         return_sequences=True,
    #         # kernel_regularizer=regularizers.l2(l2_lstm),
    #         bias_initializer=keras.initializers.Constant(0.1)),
    #         name='Bi-LSTM-{}'.format(i)
    #     )
    #
    #     model.add(bi_lstm_layer)

        # maxpool = keras.layers.MaxPooling1D()
        # model.add(maxpool)

        # attlstm = AttentionSeq2Seq(input_dim=200, input_length=100, hidden_dim=_hidden_units, output_length=100,
        #                            output_dim=200)
        # model.add(attlstm)

    for i in range(n_dense):
        '''Batch Normalization'''
        # if bn:
        #     bn = keras.layers.BatchNormalization()
        #     model.add(bn)

        # dropout_layer = keras.layers.Dropout(dropout_rate_dense)
        # model.add(dropout_layer)

        dense = keras.layers.Dense(dense_units,
                                   activation=keras.activations.relu,
                                   # kernel_regularizer=regularizers.l2(l2_dense)
                                   )
        model.add(dense)

    '''CRF'''
    crf = CRF(len(chunk_tags), sparse_target=True)
    model.add(crf)

    # model.compile('adam', loss=categorical_crossentropy, metrics=['accuracy'])
    model.compile('adam', loss=crf_loss, metrics=[crf_accuracy])
    if train:
        return model, (train_x, train_y)
    else:
        return model, (vocab, chunk_tags)

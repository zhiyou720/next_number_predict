#!/usr/bin/env python
# coding: utf-8
"""
@File     :data_helper.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/11
@Desc     : 
"""
import pickle
import itertools
import numpy as np
from collections import Counter
from pre_process_data_loader import OriginData


class DataLoader:
    def __init__(self, path, time_point_cycle=288, diy_feature=False, plot_heat_map=False,
                 data_to_predict='A', seq_len=10, class_num=10, one_class_set=None,
                 train=False, vocab_path='./model/vocab_config.pkl'):

        if class_num != 10 or one_class_set:
            n = len(one_class_set)
            if n == 0:
                n = 1
            pre_class = 10 - n + 1
            if class_num != pre_class:
                print('设置的分类数量不正确！')
                print('将 {} 设置为一类, 总共有 {} 个类别'.format(' '.join([str(x) for x in one_class_set]),
                                                      pre_class))

                class_num = pre_class

                print('自动调整分类数量，如果与预期不符，请调整配置文件')

        self.origin_data_set = OriginData(path=path,
                                          time_point_cycle=time_point_cycle,
                                          diy_feature=diy_feature)
        print('原始数据一共有{}条'.format(len(self.origin_data_set.year)))

        if plot_heat_map:
            self.origin_data_set.plot_data()

        print('开始进行batch和字典制作（读取）...')

        self.diy_feature = diy_feature

        self.data_to_predict = data_to_predict

        self.vocab_path = vocab_path

        self.seq_len = seq_len
        self.one_class_set = one_class_set
        self.class_num = class_num

        self.train = train
        self.x_train, self.y_train, self.vocabulary = self.load_data()

    def build_features(self):
        feature = []

        for i in range(len(self.origin_data_set.year)):
            tmp = [self.origin_data_set.year[i],
                   self.origin_data_set.month[i],
                   self.origin_data_set.week[i],
                   self.origin_data_set.day[i],
                   'tp' + str(self.origin_data_set.time_point[i])]

            if self.diy_feature:
                tmp += self.origin_data_set.diy_feature_set[i]
            feature.append(tmp)
        return feature

    def build_labels(self):
        if self.data_to_predict == 'A':
            return self.origin_data_set.data_a
        elif self.data_to_predict == 'B':
            return self.origin_data_set.data_b
        elif self.data_to_predict == 'DIFF':
            return self.origin_data_set.data_diff
        elif self.data_to_predict == 'DIFF_ABS':
            return self.origin_data_set.data_diff_abs
        else:
            print('请设置正确的要预测的值')
            raise ValueError

    def build_seq_data_set(self):

        """
        Load raw data and give x_train and y_train
        :return:
        """

        ptr = 0
        x_train = []
        y_train = []

        features = self.build_features()
        labels = self.build_labels()

        while ptr + self.seq_len < len(self.origin_data_set.year[:2000]):
            delta = 0
            tmp_x = []
            tmp_y = [0 for x in range(self.class_num)]
            p = 1
            while p <= self.seq_len:
                stack = features[ptr + delta] + list(labels[ptr + delta])
                tmp_x += stack
                delta += 1
                p += 1

            print(tmp_x)

            x_train.append(tmp_x)

            if self.one_class_set:

                if int(labels[ptr + self.seq_len]) in self.one_class_set:
                    tmp_y[int(self.one_class_set[0])] = 1
                else:
                    tmp_y[int(labels[ptr + self.seq_len])] = 1  # 10 classification
            else:
                tmp_y[int(labels[ptr + self.seq_len])] = 1  # 10 classification
            y_train.append(tmp_y)
            ptr += 1
        return x_train, y_train

    def load_data(self):
        """
        Loads and preprocessed data for the dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and pre-process data
        features, labels = self.build_seq_data_set()

        if self.train:
            vocabulary, vocabulary_inv = self.build_vocab(features)
            with open(self.vocab_path, 'wb') as out_p:
                pickle.dump(vocabulary, out_p)
        else:
            with open(self.vocab_path, 'rb') as inp:
                vocabulary = pickle.load(inp)

        x, y = self.build_input_data(features, labels, vocabulary)
        return [x, y, vocabulary]

    @staticmethod
    def build_vocab(sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # print(word_counts)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

        vocabulary.update(unk=len(vocabulary))

        return vocabulary, vocabulary_inv

    @staticmethod
    def build_input_data(sentences, labels, vocabulary):
        """
        Maps sentences and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
        y = np.array(labels)

        return [x, y]


if __name__ == '__main__':
    _d = DataLoader('./data/stage_2_data.csv', train=True)
    _, c, v, = _d.x_train, _d.y_train, _d.vocabulary
    print(_)

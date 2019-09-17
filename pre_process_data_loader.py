#!/usr/bin/env python
# coding: utf-8
"""
@File     :data_helper.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/16
@Desc     :
"""

from tools.dataio import load_txt_data


class OriginData:

    def __init__(self, path: str, time_point_cycle=None, diy_feature=False):
        self.raw_data = load_txt_data(path)
        self.base_feature_num = 5
        try:
            self.time_point_cycle = int(time_point_cycle)
        except ValueError:
            print('时间循环必须是数字!')
            raise ValueError
        except TypeError:
            if type(None) == type(time_point_cycle):
                self.time_point_cycle = time_point_cycle
            else:
                print('时间循环必须是数字!')
                raise TypeError

        self.date = []
        self.year = []
        self.month = []
        self.day = []
        self.time_point = []
        self.week = []
        self.data_a = []
        self.data_b = []
        self.data_diff = []
        self.data_diff_abs = []
        self.load_data()
        self.check_data()

        self.diy_feature_set = []

        if diy_feature:
            self.diy_feature()

    def load_data(self):
        tp = 1
        for i in range(len(self.raw_data)):
            if i == 0:
                continue
            item = self.raw_data[i].split(',')

            self.date.append(item[0])

            self.year.append(item[0][:4])
            self.month.append(item[0][4:6])
            self.day.append(item[0][6:])

            if self.time_point_cycle:
                if tp > self.time_point_cycle:
                    tp = 1
                self.time_point.append(tp)
                tp += 1
            else:
                self.time_point.append(int(item[1][:-1]))

            self.week.append(item[1][-1])

            self.data_a.append(item[2])
            self.data_b.append(item[3])
            self.data_diff.append(str(int(item[2]) - int(item[3])))
            self.data_diff_abs.append(item[4])

    def check_data(self):
        if len(self.date) == len(self.week) == len(self.data_a) == len(self.data_b) == len(self.data_diff_abs):
            pass
        else:
            print('原始数据表格有留白, 请填写完整')
            raise ValueError

    def diy_feature(self):

        diy_feature_num = len(self.raw_data[1]) - self.base_feature_num
        if diy_feature_num <= 0:
            print('打开diy feature开关的同时，请在源数据文件中增加新的一列')
            raise ValueError

        for raw in self.raw_data:
            self.diy_feature_set.append(raw[self.base_feature_num:])

    def build_attention_map(self, time_point, data):

        _cycle = max(self.time_point)

        data_map = [[0 for x in range(_cycle)] for y in range(10)]

        for i in range(len(time_point)):
            data_map[int(data[i])][int(time_point[i]) - 1] += 1
        return data_map

    @staticmethod
    def transpose(matrix):
        new_matrix = []
        for i in range(len(matrix[0])):
            matrix1 = []
            for j in range(len(matrix)):
                matrix1.append(matrix[j][i])
            new_matrix.append(matrix1)
        return new_matrix

    def plot_data(self):

        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt

        data_map_a = np.array(self.build_attention_map(self.time_point, self.data_a))
        data_map_b = np.array(self.build_attention_map(self.time_point, self.data_b))
        data_map_diff = np.array(self.build_attention_map(self.time_point, self.data_diff))
        data_map_diff_abs = np.array(self.build_attention_map(self.time_point, self.data_diff_abs))

        # data_map_a_t = self.transpose(data_map_a)
        # data_map_b_t = self.transpose(data_map_b)
        # data_map_diff_t = self.transpose(data_map_diff)
        # data_map_diff_abs_t = self.transpose(data_map_diff_abs)

        # for i in range(len(data_map_a_t)):
        #     line_a = data_map_a_t[i]
        #     max_freq_a = max(line_a)
        #     sum_freq_a = sum(line_a)
        #     print('{}\t{}\t{}/{}'.format(i, line_a.index(max_freq_a), max_freq_a, sum_freq_a))
        #     print('In time point {}\time_point{} always appear\time_point{}/{}'.format(i, line_a.index(max_freq_a), max_freq_a, sum_freq_a))
        #     print('{} always appear\time_point{}/{}'.format(line_a.index(max_freq_a), max_freq_a, sum_freq_a))
        #
        #     line_b = data_map_b_t[i]
        #     max_freq_b = max(line_b)
        #     sum_freq_b = sum(line_b)
        #     print('{}\t{}\t{}/{}'.format(i, line_a.index(max_freq_b), max_freq_b, sum_freq_b))
        #
        #     line_diff = data_map_diff_t[i]
        #     max_freq_diff = max(line_diff)
        #     sum_freq_diff = sum(line_diff)
        #     print('{}\t{}\t{}/{}'.format(i, line_diff.index(max_freq_diff), max_freq_diff, sum_freq_diff))
        #
        #     line_diff_abs = data_map_diff_abs_t[i]
        #     max_freq_diff_abs = max(line_diff_abs)
        #     sum_freq_diff_abs = sum(line_diff_abs)
        #     print(
        #         '{}\t{}\t{}/{}'.format(i, line_diff_abs.index(max_freq_diff_abs), max_freq_diff_abs, sum_freq_diff_abs))
        sns.set()
        plt.figure(figsize=(16, 9))
        plt.subplot(221)
        ax1 = sns.heatmap(data_map_a, center=0)
        plt.title('DATA A')
        plt.subplot(222)
        ax2 = sns.heatmap(data_map_b, center=0)
        plt.title('DATA B')
        plt.subplot(223)
        ax3 = sns.heatmap(data_map_diff, center=0)
        plt.title('DATA DIFF')
        plt.subplot(224)
        ax4 = sns.heatmap(data_map_diff_abs, center=0)
        plt.title('DATA DIFF ABS')
        plt.show()


if __name__ == '__main__':
    _ = OriginData('./data/stage_2_data.csv')
    _.load_data()
    _.plot_data()
    # cycle = 288
    # _time, _a, _b, _diff, _day, _week = build_data_set2(cycle)
    # save_data_to_csv(_a, _day, _week, _time)
    # _data = _a
    # build_attention_map(_time, _data, cycle)
    # plot_data(_time, _data, cycle)

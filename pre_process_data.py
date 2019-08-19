#!/usr/bin/env python
# coding: utf-8
"""
@File     :data_helper.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/16
@Desc     :
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tools.dataio import load_txt_data, save_txt_file
import random


def build_data_set():
    """
    build time step data
    :return:
    """
    raw_data = load_txt_data('./data/3.9_8.13.csv')
    time_step = []
    day = []
    week = []
    a = []
    b = []
    diff = []
    t = 1
    i = 0
    for item in raw_data:
        raw = item.split(',')
        # print(i)
        i += 1
        if raw:
            a.append(raw[2])
            b.append(raw[5])

            day.append(raw[0][6:8])
            week.append(raw[1])

            diff.append(raw[10])
            if t > 288:
                t = 1
            time_step.append(t)
            t += 1

    return time_step, a, b, diff, day, week


def build_attention_map(t, d):
    data_map = [[0 for x in range(288)] for y in range(10)]

    for i in range(len(t)):
        data_map[int(d[i])][int(t[i]) - 1] += 1
    return data_map


def plot_data(t, d):
    # time_stap = [str(x + 1) for x in range(288)]
    # data = [str(x) for x in range(10)]
    # data_map = np.array(build_attention_map(t, d))
    #
    # fig, ax = plt.subplots()
    # im = ax.imshow(data_map)
    #
    # ax.set_xticks(np.arange(len(time_stap)))
    # ax.set_yticks(np.arange(len(data)))
    #
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #
    # for i in range(len(data)):
    #     for j in range(len(time_stap)):
    #         text = ax.text(j, i, data_map[i, j], ha="center", va="center", color="w")
    #
    # ax.set_title("data")
    # # fig.tight_layout()
    # plt.show()

    data_map = np.array(build_attention_map(t, d))
    sns.set()
    ax = sns.heatmap(data_map, center=0)
    plt.show()


def save_data_to_csv(label, day, week, time):
    """
    label, day(date), week, time
    :param label:
    :param day:
    :param week:
    :param time:
    :return:
    """
    data = []
    for i in range(len(label)):
        raw = '{},{},{},{}'.format(label[i], day[i], week[i], time[i])
        data.append(raw)
    save_txt_file(data, './data/svm_data1.csv')


if __name__ == '__main__':
    _time, _a, _b, _diff, _day, _week = build_data_set()
    save_data_to_csv(_a, _day, _week, _time)
    build_attention_map(_time, _a)
    plot_data(_time, _diff)

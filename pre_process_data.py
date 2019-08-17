#!/usr/bin/env python
# coding: utf-8
"""
@File     :data_helper.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/16
@Desc     :
"""
import matplotlib.pyplot as plt
from tools.dataio import load_txt_data, save_txt_file
import random


def build_data_set():
    """
    build time step data
    :return:
    """
    raw_data = load_txt_data('./data/raw_data.csv')
    raw_data += load_txt_data('./data/new_raw_data.txt')
    time_step = []
    label = []

    rnd = random.randint(6, 20)

    for i in range(len(raw_data)):
        # time_step.append(i % (288 * 14))
        time_step.append(i)
        y = raw_data[i]
        # label.append(y)
        if 0 < int(y) <= 3:
            label.append(0)
        else:
            label.append(1)

        # if (i + 1) % rnd == 0:
        # # if (i + 1) % (288 * 7) == 0 and i != 0:
        #     time_step.append(' ')
        #     label.append(' ')
        #     rnd = random.randint(6, 20) + i

    data = []

    for i in range(len(time_step)):
        if time_step[i] == ' ':
            data.append(None)
        else:
            data.append('{} O_{}'.format(time_step[i], label[i]))

    return data, time_step, label


def plot_data(x, y):
    plt.plot(x[:576], y[:576], 'ro')
    plt.show()


if __name__ == '__main__':
    _d, _, __ = build_data_set()
    save_txt_file(_d, './data/format_data.txt')
    print(len(_))
    plot_data(_, __)

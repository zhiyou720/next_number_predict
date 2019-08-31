#!/usr/bin/env python
# coding: utf-8
"""
@File     :dataio.py
@Copyright: zhiyou720
@Contact  : zhiyou720@gmail.com
@Date     :2019/8/11
@Desc     : 
"""
import os


def load_txt_data(path):
    """
    This func is used to reading txt file
    :param path: path where file stored
    :type path: str
    :return: string lines in file in a list
    :rtype: list
    """
    if type(path) != str:
        raise TypeError
    res = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            res.append(line.strip())

    return res


def save_txt_file(data, path, end='\n', encoding='utf-8'):
    """
    This func is used to saving data to txt file
    support data type:
    list: Fully support
    dict: Only save dict key
    str: will save single char to each line
    tuple: Fully support
    set: Fully support
    :param encoding:
    :param data: data
    :param path: path to save
    :type path: str
    :param end:
    :type end: str
    :return: None
    """
    if type(data) not in [list, dict, str, tuple, set] or type(path) != str:
        raise TypeError

    remove_old_file(path)

    with open(path, 'a', encoding=encoding) as f:
        for item in data:
            f.write(str(item) + end)


def remove_old_file(path):
    """
    :param path:
    :type path: str
    :return:
    """
    if check_dir(path):
        os.remove(path)


def check_dir(path):
    """
    check dir exists
    :param path:
    :type path:str
    :return:
    :rtype: bool
    """
    return os.path.exists(path)

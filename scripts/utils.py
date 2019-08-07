# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/26 14:45
   desc: the project
"""
import numpy as np


def one_hot(y: np.ndarray):
    """
    对行向量或者列向量进行one-hot编码
    :param y:
    :return:
    """
    if len(y.shape) > 2:
        print("cannot encode data, because data shape larger than 2")
    else:
        y = y.reshape((-1, 1))  # 转为列向量
        values_all = set()
        for item in y:
            values_all.add(int(item))
        values_all = list(values_all)
        y_encoded = []
        for i in range(y.shape[0]):
            v = np.zeros((len(values_all)))
            v[values_all.index(int(y[i]))] = 1
            y_encoded.append(v)
        return np.array(y_encoded, dtype=np.int)
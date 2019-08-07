# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/26 15:20
   desc: the project
"""
import numpy as np


def mse(y, y_true):
    """
    计算MSE损失
    :param y:
    :param y_true:
    :return:
    """
    n = y.shape[0]
    loss = np.sum((y_true - y)**2) / n
    return loss


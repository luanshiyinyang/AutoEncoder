# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/26 15:06
   desc: 本模块放置激活函数机器倒数
"""
import numpy as np


def sigmoid(x):
    """
    Sigmoid激活函数
    :param x:
    :return:
    """
    s = 1.0 / (1.0 + np.exp(-x))
    return s


def sigmoid_grad(x):
    """
    sigmoid激活函数的导数
    :param x:
    :return:
    """
    grad = sigmoid(x) * (1-sigmoid(x))
    return grad


def relu(x):
    """
    Relu激活函数
    :param x:
    :return:
    """
    s = np.maximum(x, 0.0)
    return s


def relu_grad(x):
    """
    Relu激活函数导数
    :param x:
    :return:
    """
    z = relu(x)
    z[z > 0] = 1
    return z

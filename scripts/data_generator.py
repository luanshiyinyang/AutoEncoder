# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/26 12:29
   desc: 本模块用于数据读取并规范化
"""
import numpy as np
import os


class MNIST(object):
    def __init__(self, load_path):
        """
        作为比较经典的手写数据集之一
        这里已经预先处理了官方数据集为npy格式
        该数据集包含60000张训练数据和10000张测试数据
        """
        self.path = load_path

    def get_train_data(self):
        """
        读取训练数据，返回numpy矩阵
        :return: x_train
        :return: y_train
        """
        x_train = np.load(os.path.join(self.path, 'x_train.npy'))
        y_train = np.load(os.path.join(self.path, 'y_train.npy'))
        return x_train, y_train

    def get_test_data(self):
        """
        读取测试数据，返回numpy矩阵
        :return: x_test
        :return: y_test
        """
        x_test = np.load(os.path.join(self.path, 'x_test.npy'))
        y_test = np.load(os.path.join(self.path, 'y_test.npy'))
        return x_test, y_test


class USPS(object):
    def __init__(self, load_path):
        """
        USPS为美国邮政手写数据集
        数据已经处理为npy格式
        7291个训练集和2007个测试集
        """
        self.path = load_path

    def get_train_data(self):
        """
        读取训练数据，返回numpy矩阵
        :return: x_train
        :return: y_train
        """
        x_train = np.load(os.path.join(self.path, 'x_train.npy'))
        y_train = np.load(os.path.join(self.path, 'y_train.npy'))
        return x_train, y_train

    def get_test_data(self):
        """
        读取测试数据，返回numpy矩阵
        :return: x_test
        :return: y_test
        """
        x_test = np.load(os.path.join(self.path, 'x_test.npy'))
        y_test = np.load(os.path.join(self.path, 'y_test.npy'))
        return x_test, y_test


class Semeion(object):
    def __init__(self, load_path):
        """
        Semeion是手写集比较经典的
        这里预先处理了官方数据
        包含1195个训练数据和398个测试数据
        :param load_path:
        """
        self.path = load_path

    def get_train_data(self):
        """
        读取训练数据，返回numpy矩阵
        :return: x_train
        :return: y_train
        """
        x_train = np.load(os.path.join(self.path, 'x_train.npy'))
        y_train = np.load(os.path.join(self.path, 'y_train.npy'))
        return x_train, y_train

    def get_test_data(self):
        """
        读取测试数据，返回numpy矩阵
        :return: x_test
        :return: y_test
        """
        x_test = np.load(os.path.join(self.path, 'x_test.npy'))
        y_test = np.load(os.path.join(self.path, 'y_test.npy'))
        return x_test, y_test


if __name__ == '__main__':
    mnist = Semeion('../data/Semeion')
    x_train, _ = mnist.get_train_data()
    x_test, _ = mnist.get_test_data()
    print(x_train.shape, x_test.shape)

# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/26 14:44
   desc: the project
"""
import numpy as np
from tqdm import tqdm
from activation_func import sigmoid, sigmoid_grad
from loss_function import mse


class AutoEncoder(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        """
        构建静态的BP全连接神经网络
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # 初始化权重
        self.w_input_to_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.hidden_bias = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, 1))

        self.w_hidden_to_output = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        self.output_bias = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, 1))

    def fit(self, x_train_, y_train_, epochs=100, lr=0.1):
        """
         模型训练
        :param x_train_:
        :param y_train_:
        :param epochs:
        :param lr:
        :return:
        """

        for epoch in range(epochs):

            for i in range(x_train_.shape[0]):
                x = x_train_[i].reshape(-1, 1)  # (4, 1)
                y = y_train_[i].reshape(-1, 1)  # (3, 1)

                # forward前向传播
                hidden_layer_input = np.dot(self.w_input_to_hidden, x).reshape(-1, 1) - self.hidden_bias
                hidden_layer_output = sigmoid(hidden_layer_input)  # 激活为非线性

                output_layer_input = np.dot(self.w_hidden_to_output, hidden_layer_output).reshape(-1, 1) - self.output_bias
                output_layer_output = sigmoid(output_layer_input)
                # error计算误差
                # backward反向传播
                theta = (y - output_layer_output) * sigmoid_grad(output_layer_input)  # (3, 1)
                self.w_hidden_to_output += lr * np.dot(theta, hidden_layer_output.T)
                self.output_bias += -lr * theta

                beta = np.dot(self.w_hidden_to_output.T, theta) * sigmoid_grad(hidden_layer_input)
                self.w_input_to_hidden += lr * np.dot(beta, x.T)
                self.hidden_bias += -lr * beta

            loss = self.evaluate(x_train_, y_train_)
            # print("Epoch train {}:Train loss:{:.4f}".format(epoch, loss))

    def evaluate(self, x_test_, y_test_):
        """
        验证集评估
        :return:
        """
        x_test_ = x_test_.reshape(x_test_.shape[0], -1)
        loss = 0
        for i in range(x_test_.shape[0]):
            x = x_test_[i].reshape(-1, 1)
            y = y_test_[i].reshape(-1, 1)
            x = sigmoid(np.dot(self.w_input_to_hidden, x).reshape(-1, 1) - self.hidden_bias)
            pred = sigmoid(np.dot(self.w_hidden_to_output, x).reshape(-1, 1) - self.output_bias)
            loss += mse(y, pred)

        return loss / x_test_.shape[0]

    def encode(self, x_input):
        """
        预测新数据
        :param x_input:
        :return:
        """
        x_test_ = x_input.reshape(x_input.shape[0], -1)
        rst = []
        for i in range(x_test_.shape[0]):
            x = x_test_[i]
            x = sigmoid(np.dot(self.w_input_to_hidden, x).reshape(-1, 1) - self.hidden_bias)
            rst.append(np.squeeze(x, axis=-1))
        return np.array(rst).astype('float32')



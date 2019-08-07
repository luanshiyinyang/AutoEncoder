# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/31 23:11
   desc: 利用keras构建模型的尝试
"""
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD


def get_model(encoding_dim=10):
    """
    建立模型
    :param encoding_dim:
    :return:
    """
    input_layer = Input(shape=(784, ))
    encoded = Dense(encoding_dim, activation='sigmoid', use_bias=True)(input_layer)
    decoded = Dense(784, activation='sigmoid', use_bias=False)(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    optimizer = SGD(momentum=0.9, decay=1e-5)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    return autoencoder, encoder

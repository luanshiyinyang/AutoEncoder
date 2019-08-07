# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/5/27 15:02
   desc: 在mnist数据集上搜索不同降维目标的表现
"""
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from data_generator import MNIST
from AutoEncoder import AutoEncoder


def test_pca(x_train_, y_train_, x_test_, y_test_):
    """
    测试pca效果
    :return:
    """
    rst = []
    # 数据展平
    x_train_ = x_train_.reshape(x_train_.shape[0], -1)
    x_test_ = x_test_.reshape(x_test_.shape[0], -1)
    # 测试数据一半用于svc训练，一半用于测试
    x_test_train = x_test_[:5000]
    y_test_train = y_test_[:5000]
    x_test_test = x_test_[5000:]
    y_test_test = y_test_[5000:]
    # k对应不同降维目标
    for k in range(10, 200, 10):
        # 训练pca
        pca = PCA(n_components=k)
        pca.fit(x_train_)
        # 训练svc分类器
        svc = SVC(gamma='scale')
        svc.fit(pca.transform(x_test_train), y_test_train)
        # 测试分类器
        y_pred = svc.predict(pca.transform(x_test_test))
        accuracy = accuracy_score(y_test_test, y_pred)
        print(accuracy)
        rst.append(accuracy)
    return rst


def test_ae(x_train_, y_train_, x_test_, y_test_):
    """
    AE测试
    """
    rst = []
    # 数据展平
    x_train_ = x_train_.reshape(x_train_.shape[0], -1)
    x_test_ = x_test_.reshape(x_test_.shape[0], -1)
    # 测试数据一半用于svc训练，一半用于测试
    x_test_train = x_test_[:5000]
    y_test_train = y_test_[:5000]
    x_test_test = x_test_[5000:]
    y_test_test = y_test_[5000:]
    # k对应不同降维目标
    for k in range(10, 200, 10):
        # 训练pca
        ae = AutoEncoder(28*28, k, 28*28)
        ae.fit(x_train_, x_train_)
        # 训练svc分类器
        svc = SVC(gamma='scale')
        svc.fit(ae.encode(x_test_train), y_test_train)
        # 测试分类器
        y_pred = svc.predict(ae.encode(x_test_test))
        accuracy = accuracy_score(y_test_test, y_pred)
        print(accuracy)
        rst.append(accuracy)
    return rst


if __name__ == '__main__':
    # 获取数据集
    mnist = MNIST('../data/MNIST')
    x_train, y_train = mnist.get_train_data()
    x_test, y_test = mnist.get_test_data()
    history1 = test_pca(x_train.copy()[:10000]/255., y_train.copy()[:10000], x_test.copy()/255., y_test.copy())
    history2 = test_ae(x_train.copy()[:10000]/255., y_train.copy()[:10000], x_test.copy()/255., y_test.copy())
    # 具体绘图在ipynb里面


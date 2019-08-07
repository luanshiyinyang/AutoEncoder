# -*-coding:utf-8-*-
"""author: Zhou Chen
   datetime: 2019/6/1 11:21
   desc: the project
"""
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from model import get_model
from data_generator import MNIST, USPS, Semeion


def test_pca_mnist(x, y, x_test, y_test):
    """
    :param x
    :param y
    :param x_test
    :param y_test
    :return:
    """
    x = x.reshape(x.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # 构建并训练pca
    pca = PCA(n_components=100)
    pca.fit(x)

    x_svm_train = x_test[:5000].copy()
    y_svm_train = y_test[:5000].copy()
    x_svm_test = x_test[5000:].copy()
    y_svm_test = y_test[5000:].copy()
    # 建立分类器
    svc = SVC(gamma='scale')
    # 训练分类器
    svc.fit(pca.transform(x_svm_train), y_svm_train)
    # 使用分类器
    y_pred = svc.predict(pca.transform(x_svm_test))
    return accuracy_score(y_svm_test, y_pred)


def test_ae_mnist(x, y, x_test, y_test):
    """
    AE测试
    :param x:
    :param y:
    :param x_test:
    :param y_test:
    :return:
    """
    x = x.reshape(x.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    # 构建并训练ae
    ae, encoder = get_model(100)
    ae.fit(x, x, epochs=50, shuffle=False, verbose=0)

    x_svm_train = x_test[:5000].copy()
    y_svm_train = y_test[:5000].copy()
    x_svm_test = x_test[5000:].copy()
    y_svm_test = y_test[5000:].copy()
    # 建立分类器
    svc = SVC(gamma='scale')
    # 训练分类器
    svc.fit(encoder.predict(x_svm_train), y_svm_train)
    # 使用分类器
    y_pred = svc.predict(encoder.predict(x_svm_test))
    return accuracy_score(y_svm_test, y_pred)


def test_pca_usps(x, y, x_test, y_test):
    """
    :param x
    :param y
    :param x_test
    :param y_test
    :return:
    """
    x = x.reshape(x.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # 构建并训练pca
    pca = PCA(n_components=100)
    pca.fit(x)

    x_svm_train = x_test[:1000].copy()
    y_svm_train = y_test[:1000].copy()
    x_svm_test = x_test[1000:].copy()
    y_svm_test = y_test[1000:].copy()
    # 建立分类器
    svc = SVC(gamma='scale')
    # 训练分类器
    svc.fit(pca.transform(x_svm_train), y_svm_train.reshape(-1))
    # 使用分类器
    y_pred = svc.predict(pca.transform(x_svm_test))
    return accuracy_score(y_svm_test.reshape(-1), y_pred.reshape(-1))


def test_ae_usps(x, y, x_test, y_test):
    """
    AE测试
    :param x:
    :param y:
    :param x_test:
    :param y_test:
    :return:
    """
    x = x.reshape(x.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    # 构建并训练ae
    ae, encoder = get_model(100)
    ae.fit(x, x, epochs=1000, shuffle=False, verbose=0)

    x_svm_train = x_test[:1000].copy()
    y_svm_train = y_test[:1000].copy()
    x_svm_test = x_test[1000:].copy()
    y_svm_test = y_test[1000:].copy()
    # 建立分类器
    svc = SVC(gamma='scale')
    # 训练分类器
    svc.fit(encoder.predict(x_svm_train), y_svm_train.reshape(-1))
    # 使用分类器
    y_pred = svc.predict(encoder.predict(x_svm_test))
    return accuracy_score(y_svm_test, y_pred)


def test_pca_semeion(x, y, x_test, y_test):
    """
    :param x
    :param y
    :param x_test
    :param y_test
    :return:
    """
    x = x.reshape(x.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # 构建并训练pca
    pca = PCA(n_components=100)
    pca.fit(x)

    x_svm_train = x_test[:200].copy()
    y_svm_train = y_test[:200].copy()
    x_svm_test = x_test[200:].copy()
    y_svm_test = y_test[200:].copy()
    # 建立分类器
    svc = SVC(gamma='scale')
    # 训练分类器
    svc.fit(pca.transform(x_svm_train), y_svm_train.reshape(-1))
    # 使用分类器
    y_pred = svc.predict(pca.transform(x_svm_test))
    return accuracy_score(y_svm_test.reshape(-1), y_pred.reshape(-1))


def test_ae_semeion(x, y, x_test, y_test):
    """
    AE测试
    :param x:
    :param y:
    :param x_test:
    :param y_test:
    :return:
    """
    x = x.reshape(x.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    # 构建并训练ae
    ae, encoder = get_model(100)
    ae.fit(x, x, epochs=20, shuffle=False, verbose=0)

    x_svm_train = x_test[:200].copy()
    y_svm_train = y_test[:200].copy()
    x_svm_test = x_test[200:].copy()
    y_svm_test = y_test[200:].copy()
    # 建立分类器
    svc = SVC(gamma='scale')
    # 训练分类器
    svc.fit(encoder.predict(x_svm_train), y_svm_train.reshape(-1))
    # 使用分类器
    y_pred = svc.predict(encoder.predict(x_svm_test))
    return accuracy_score(y_svm_test, y_pred)


def test_mnist():
    mnist = MNIST('../data/MNIST')
    x_train, y_train = mnist.get_train_data()
    x_train = x_train / 255.
    x_test, y_test = mnist.get_test_data()
    x_test = x_test / 255.
    x_train, x_t, y_train, y_t = train_test_split(x_train, y_train, test_size=0, random_state=2019)  # 打乱数据
    x_test, x_t, y_test, y_t = train_test_split(x_test, y_test, test_size=0, random_state=2019)
    print(test_pca_mnist(x_train[:10000], y_train[:10000], x_test, y_test))
    print(test_ae_mnist(x_train[:10000], y_train[:10000], x_test, y_test))


def test_usps():
    usps = USPS('../data/USPS')
    x_train, y_train = usps.get_train_data()
    x_train = x_train / 255.
    x_test, y_test = usps.get_test_data()
    x_test = x_test / 255.
    x_train, x_t, y_train, y_t = train_test_split(x_train, y_train, test_size=0, random_state=2019)  # 打乱数据
    x_test, x_t, y_test, y_t = train_test_split(x_test, y_test, test_size=0, random_state=2019)
    print(test_pca_usps(x_train, y_train, x_test, y_test))
    print(test_ae_usps(x_train, y_train, x_test, y_test))


def test_semeion():
    usps = USPS('../data/Semeion')
    x_train, y_train = usps.get_train_data()
    x_train = x_train / 255.
    x_test, y_test = usps.get_test_data()
    x_test = x_test / 255.
    x_train, x_t, y_train, y_t = train_test_split(x_train, y_train, test_size=0, random_state=2019)  # 打乱数据
    x_test, x_t, y_test, y_t = train_test_split(x_test, y_test, test_size=0, random_state=2019)
    print(test_pca_semeion(x_train, y_train, x_test, y_test))
    print(test_ae_semeion(x_train, y_train, x_test, y_test))


if __name__ == '__main__':
    # test_mnist()  # 0.95, 0.96左右
    # test_usps()
    test_semeion()  # 增广训练才能达到更好效果

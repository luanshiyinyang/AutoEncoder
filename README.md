# 自编码器


## 说明
- 人工智能课程的大作业，实现自编码器进行图像特征提取。
- 源码包括梯度下降均是利用矩阵运算包numpy从底层运算开始实现的（利用BP神经网络）。
- 在MNIST、USPS、Semeion三个手写数据集（本项目提供数据集的npy文件）上进行训练及验证。


## 效果
- 通过将用PCA无监督提取的特征向量和AutoEncoder有监督提取的同维特征向量送入SVM分类器，对比分类准确率。（下图摘自完成的论文）
  - ![](/assets/rst.png)


## 项目说明
- data/ 放置数据集
    - MNIST/ 放置处理好的28*28的mnist数据集的训练和测试数据的numpy矩阵文件
    - USPS/ 放置处理好的28*28的USPS数据集的训练和测试数据的numpy矩阵文件
    - Semeion/ 放置处理好的28*28的Semeion数据集的训练和测试数据的numpy矩阵文件
- scripts/ 放置Python脚本
    - activation_func.py 使用的激活函数及其梯度计算
    - AutoEncoder.py 基于实验的BP网络代码修改得到的自编码器模型
    - data_generator.py 数据生成器，获取不同数据集的数据
    - loss_function.py 损失函数
    - model.py 使用第三方库构建的多层自编码器，前期用于对比测试的
    - test.py 循环搜索mnist集上不同降维目标下降维的数据价值
    - test.ipynb 同上目的的jupyter脚本，由于py脚本非交互性，且每次运行较久，换用jupyter
    - test_accuray.py 训练并得到不同数据集的最好准确率
    - utils.py 常用的工具库，只写了一个onehot编码函数
- requirements.txt 使用的第三方包


## 补充说明
- 具体完成的论文不提供，需要请私戳[我](mailto:luanshiyinyang@gmail.com)。
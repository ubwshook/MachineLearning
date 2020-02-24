"""
简单实现感知机，对数据集iris进行分类
"""

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


class Perceptron():
    # 初始化w，b，学习率
    def __init__(self, lr=0.01):
        self.w = 0
        self.b = 0
        self.lr = lr

    # 训练函数
    def fit(self, X_train, y_train):
        w = np.zeros(len(X_train[0]))
        b = 0
        lr = self.lr
        all_true = False
        # 只要存在误分类点就继续循环
        while not all_true:
            all_true = True
            for i in range(len(X_train)):
                # 误分类条件
                if y_train[i] * (np.dot(w, X_train[i]) + b) <= 0:
                    all_true = False
                    # 更新w，b
                    w += lr * y_train[i] * X_train[i]
                    b += lr * y_train[i]

        self.w = w
        self.b = b

    # 预测函数
    def predict(self, X_test):
        res = []
        for i in range(len(X_test)):
            if np.dot(self.w, X_test[i]) + self.b <= 0:
                res.append(-1)
            else:
                res.append(1)
        return np.array(res)

    # 评价函数
    def score(self, y_predict, y_test):
        return np.mean(y_predict == y_test)


def make_data():
    """
    生成所需要的数据集，这里使用的是iris数据集。
    Iris数据集在模式识别研究领域应该是最知名的数据集了，有很多文章都用到这个数据集。这个数据集里一共包括150行记录，
    其中前四列为花萼长度，花萼宽度，花瓣长度，花瓣宽度等4个用于识别鸢尾花的属性，第5列为鸢尾花的类别（包括Setosa，Versicolour，
    Virginica三类）。也即通过判定花萼长度，花萼宽度，花瓣长度，花瓣宽度的尺寸大小来识别鸢尾花的类别。
    我们使用前100个数据，也就只有前两种分类，同时我们仅使用两种属性即花萼长度、花萼宽度。
    :return: 总训数据集 train_data, 训练数据 X_train, 测试数据 X_test, 训练数据标签 y_train, y_test, 测试数据标签 feature_names
    """
    # 导入iris数据集
    iris = load_iris()

    data = iris.data
    data = data[:100]
    labels = iris.target
    labels = labels[:100]
    feature_names = iris.feature_names
    print("feature_name", feature_names)

    # 我们选择了sepal length (cm)', 'sepal width (cm)'，并且更改了labels
    train_data = data[:, 0:2]
    labels[labels == 0] = -1
    feature_names = feature_names[:2]

    # 划分数据集，75个训练数据，25个测试数据
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels)  # 划分数据集
    return train_data, X_train, X_test, y_train, y_test, feature_names


train_data, X_train, X_test, y_train, y_test, feature_names = make_data()

# 创建模型、进行训练、测试结果评分
model = Perceptron()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print("predict score: " + str(model.score(y_predict, y_test)))

# 绘制样本以及分类示意图
w = model.w
b = model.b
plt.scatter(train_data[:50, 0], train_data[:50, 1])
plt.scatter(train_data[50:-1, 0], train_data[50:-1, 1])
xx = np.linspace(train_data[:, 0].min(), train_data[:, 0].max(), 100)
yy = -(w[0] * xx + b) / w[1]
plt.plot(xx, yy)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.show()

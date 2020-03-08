"""
简单实现感知机，对数据集iris进行分类
"""

from matplotlib import pyplot as plt
import numpy as np
from dataset import get_iris_data


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


train_data, X_train, X_test, y_train, y_test, feature_names = get_iris_data()

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

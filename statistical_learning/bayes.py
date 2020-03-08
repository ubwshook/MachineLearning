"""
实现贝叶斯，对数据集iris进行分类
"""
from collections import Counter
from dataset import get_iris_data

train_data, X_train, X_test, y_train, y_test, feature_names = get_iris_data()

""" 计算先验概率 """
p_class_0 = Counter(y_train)[-1] / len(y_train)
p_class_1 = Counter(y_train)[1] / len(y_train)

""" 计算每个特征的条件概率 """
counter_x1 = Counter(row[0] for row in X_train)
counter_x2 = Counter(row[1] for row in X_train)
x1_label = [(row[0], y_train[index]) for index, row in enumerate(X_train)]
counter_x1_label = Counter(x1_label)
x2_label = [(row[1], y_train[index]) for index, row in enumerate(X_train)]
counter_x2_label = Counter(x2_label)

for key in counter_x1_label:
    counter_x1_label[key] = counter_x1_label[key] / counter_x1[key[0]]

for key in counter_x2_label:
    counter_x2_label[key] = counter_x2_label[key] / counter_x2[key[0]]

""" 使用测试数据，进行验证，计算后验概率。 """
good_count = 0
for index, row in enumerate(X_test):
    p_x1_0 = counter_x1_label[(row[0], -1)]
    p_x2_0 = counter_x2_label[(row[1], -1)]
    P_test_0 = p_class_0 * p_x1_0 * p_x2_0
    P_test_1 = p_class_1 * counter_x1_label[(row[0], 1)] * counter_x2_label[(row[1], 1)]
    if P_test_0 > P_test_1:
        predict = -1
    else:
        predict = 1

    if predict == y_test[index]:
        good_count += 1

""" 计算准确率 """
print('准确率: ' + str((good_count / len(X_test)) * 100) + '%')




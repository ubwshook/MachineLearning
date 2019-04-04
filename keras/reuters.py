"""
这次我们构建网络将路透社新闻划分为46个互斥主题，这是一个典型的“单标签，多分类”问题
"""
# from keras.datasets import reuters
import numpy as np
from keras.utils import to_categorical
from tools import vectorize_sequences
from keras import models
from keras import layers
import matplotlib.pyplot as plt


def load_local(num_words=None):
    """
    @函数功能:加载本地数据，仿照keras源码
    """
    test_split = 0.2
    seed = 113
    start_char = 1
    oov_char = 2
    index_from = 3
    skip_top = 0
    with np.load('reuters.npz') as f:
        xs, labels = f['x'], f['y']

    np.random.seed(seed)
    indices = np.arange(len(xs))
    np.random.shuffle(indices)
    xs = xs[indices]
    labels = labels[indices]

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if skip_top <= w < num_words else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = int(len(xs) * (1 - test_split))
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def show_acc(history):
    """ 绘制精度曲线 """
    plt.clf()
    history_dict = history.history
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']

    epochs = range(1, len(val_acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

    plt.show()


(train_data, train_labels), (test_data, test_labels) = load_local(num_words=10000)

""" 训练数据向量化 """
x_train = vectorize_sequences(train_data)
y_train = vectorize_sequences(test_data)

""" one-hot处理标签 """
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(train_labels)

""" 构建神经网络 
有两个中间层，每层64个隐藏单元
多分类问题，最后输出层使用softmax激活函数
"""
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

""" 编译模型 """
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

""" 留出验证集 """
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

""" 训练网络 """
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

""" 展示结果 """
show_acc(history)

"""
IMDB数据集包含来自互联网电影数据库(IMDB)的50000条严重两级分化的评论
这个脚本叫去尝试分类一条评论是正面还是负面
"""
#from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import losses, metrics, optimizers
import matplotlib.pyplot as plt
from tools import vectorize_sequences


def load_local(num_words=None):
    """
    @函数功能:加载本地数据，仿照keras源码
    """
    skip_top = 0
    seed = 113
    start_char = 1
    oov_char = 2
    index_from = 3
    with np.load('imdb.npz') as f:
        f.allow_pickle = True
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

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
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
              for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words]
              for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)


def show_loss(history):
    """ 绘制损失曲线 """
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def show_acc(history):
    """ 绘制精度曲线 """
    plt.clf()
    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']

    epochs = range(1, len(val_acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

    plt.show()


"""
@加载数据集, num_words=10000是仅保留训练数据中前10000个最常出现的单词
本应该使用: (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
但因为下载地址是国外网站经常断线，所以自己下载数据文件后，用自己的load_local方式加载数据
"""
(train_data, train_labels), (test_data, test_labels) = load_local(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

"""把标签向量化"""
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

""" 构建神经网络 
有两个中间层，每层16个隐藏单元
最后输出层使用sigmoid激活函数
"""
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

""" 编译网络 
因为是二分类问题，所以使用二元交叉熵函数作为损失函数，评价标准还是准确性
"""
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

"""留出一部分验证集"""
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

"""训练网络"""
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

"""画出评价曲线"""
show_loss(history)
show_acc(history)

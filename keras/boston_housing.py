from keras.datasets import boston_housing
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt


def load_local(path='boston_housing.npz', test_split=0.2, seed=113):
    """Loads the Boston Housing dataset.

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
        test_split: fraction of the data to reserve as test set.
        seed: Random seed for shuffling the data
            before computing the test split.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    assert 0 <= test_split < 1
    with np.load(path) as f:
        x = f['x']
        y = f['y']

    np.random.seed(seed)
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)


def build_model():
    """
    构建网络
    这个样本数据量很少，我们将使用一个非常小的网络。训练数据越少，过拟合就会越严重，而较小的网络可以降低过拟合。
    """
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    """最后一层只有一个单元，没有激活"""
    model.add(layers.Dense(1))
    """损失函数使用的均方误差，评价网络则使用的是平均绝对误差(MAE)"""
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def smooth_curve(points, factor=0.9):
    """
    将每个数据点替换为前面数据点的指数移动平均值，以得到到光滑的曲线
    :param points:
    :param factor:
    :return:
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def show_mae(average_mae_history):
    """
    绘制验证，对数据平滑，并且删除前10个数据点
    """
    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()


(train_data, train_targets), (test_data, test_targets) = load_local()

"""数据标准化: 减去平均值，再除以标准差，这样得到数据平均值为0，标准差为1"""
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

"""
这里使用K值验证法，这是对小数据量网络处理的方式
"""
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)

    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]],
                                           axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

"""计算所有伦才中MAE的平均值"""
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

"""绘制mae曲线"""
show_mae(average_mae_history)

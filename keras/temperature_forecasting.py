"""
耶拿温度预测问题，这个数据集每14分钟记录14个的气候特征如气温，气压，湿度，风向等。
我们使用多种网络模型进行温度预测
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from tools import plt_loss


def get_temperature_data():
    """
    对原始数据处理
    :return: 返回标准化后的浮点数据，和标准差
    """
    data_dir = 'E:/git_code/data/jena_climate'
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
    f = open(fname)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    print(header)
    print(len(lines))

    float_data = np.zeros((len(lines), len(header) - 1))
    for index, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]
        float_data[index, :] = values

    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std

    return float_data, std


def show_sequence(float_data):
    """
    展示序列图像
    :param float_data: 数据序列
    :return: 返回浮点数据
    """
    temp = float_data[:, 1]
    plt.plot(range(len(temp)), temp)
    plt.show()
    plt.plot(range(1440), temp[:1440])
    plt.show()


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    """
    根据参数生成数据，用于进行训练
    :param data: 浮点数据组成的原始数据
    :param lookback: 输入数据应该包括过去的多少个时间步
    :param delay: 目标应该在未来的多少个时间步
    :param min_index: data中的索引，用于确定抽取数据的范围
    :param max_index: data中的索引，用于确定抽取数据的范围
    :param shuffle: 是否打乱样本
    :param batch_size: 每批数据的样本数
    :param step: 数据采样周期。 设置为6，每小时抽取一个数据点
    :return:
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


def evaluate_naive_method():
    """
    计算一个基准精度，始终预测24小时后的温度与现在的温度相同，以下代码就计算局方绝对误差(MAE)指标来评估这种方法。
    :return:
    """
    batch_maes = []
    for step in range(VAL_STEPS):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]  # 使用最后一个采样点的温度作为预测值
        mae = np.mean(np.abs(preds - targets))  # 计算误差
        batch_maes.append(mae)
    print(np.mean(batch_maes))


def train_dense(float_data):
    """
    密集连接网络预测问题
    :param float_data: 标准化后温度数据
    :return:
    """
    model = Sequential()
    model.add(layers.Flatten(input_shape=(LOOK_BACK // STEP, float_data.shape[-1])))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                                  validation_data=val_gen, validation_steps=VAL_STEPS)
    plt_loss(history)


def train_gru(float_data):
    """
    使用GRU循环网络预测温度
    :param float_data: 标准化后温度数据
    :return:
    """
    model = Sequential()
    model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                                  validation_data=val_gen, validation_steps=VAL_STEPS)
    plt_loss(history)


def train_gru_dropout(float_data):
    """
    使用GRU循环网络并使用dropout来降低过拟合
    :param float_data: 标准化后温度数据
    :return:
    """
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                                  validation_data=val_gen, validation_steps=VAL_STEPS)
    plt_loss(history)


def train_stacking_gru(float_data):
    """
    使用GRU循环网络 + dropout + 堆叠
    :param float_data: 标准化后温度数据
    :return:
    """
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True,
                         input_shape=(None, float_data.shape[-1])))
    model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen,
                                  validation_steps=VAL_STEPS)
    plt_loss(history)


def train_bidirectional_gru(float_data):
    """
    使用双向GRU来训练网络
    :return:
    """
    model = Sequential()
    model.add(layers.Bidirectional(
    layers.GRU(32), input_shape=(None, float_data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                                  validation_data=val_gen, validation_steps=VAL_STEPS)
    plt_loss(history)


if __name__ == "__main__":
    float_data, std = get_temperature_data()  # 获取处理后的数据和标准差
    LOOK_BACK = 1440  # 输入数据应该包括过去的多少个时间步
    STEP = 6  # 数据采样周期。 设置为6，每小时抽取一个数据点
    DELAY = 144  # 目标应该在未来的多少个时间步
    BATCH_SIZE = 128  # 每批数据的样本数
    VAL_STEPS = (300000 - 200001 - LOOK_BACK) // BATCH_SIZE  # 验证数据的批次数量
    TEST_STEPS = (len(float_data) - 300001 - LOOK_BACK) // BATCH_SIZE  # 测试数据的批次数量

    # 初始化生成器
    train_gen = generator(float_data, lookback=LOOK_BACK, delay=DELAY, min_index=0,
                          max_index=200000, shuffle=True, step=STEP, batch_size=BATCH_SIZE)
    val_gen = generator(float_data, lookback=LOOK_BACK, delay=DELAY, min_index=200001,
                        max_index=300000, step=STEP, batch_size=BATCH_SIZE)
    test_gen = generator(float_data, lookback=LOOK_BACK, delay=DELAY, min_index=300001,
                         max_index=None, step=STEP, batch_size=BATCH_SIZE)

    show_sequence(float_data)

    #evaluate_naive_method()  # 计算基准误差
    #celsius_mae = 0.29 * std[1]  # 温度的绝对偏差

    #train_dense(float_data)  # 密集连接网络预测温度
    #train_gru(float_data)  # 单层GRU网络进行训练
    #train_gru_dropout(float_data)  # 带有dropout正则化得GRU
    train_stacking_gru(float_data)  # 堆叠gru训练
    train_bidirectional_gru(float_data)  # 双向GRU训练










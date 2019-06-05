"""
简单RNN原理的纯python实现、keras实现、LSTM实现
"""

import numpy as np
from keras.preprocessing import sequence
from imdb import load_local
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM
from tools import show_acc_results

"""
纯python实现一个RNN的原理
"""
timesteps = 100  # 输入序列的时间步
input_features = 32  # 输入特征空间维度
output_features = 64  # 输出特征空间维度

inputs = np.random.random((timesteps, input_features))  # 输入数据，此处仅为示意
state_t = np.zeros((output_features,))  # 出事状态为0

# 创建随机的权重矩阵
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs:  # 输入形状为(input_features,)的向量
    # 由输入和当前状态计算得到输出
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)  # 将输出保存到列表中
    state_t = output_t  # 更新网络状态

final_output_sequence = np.concatenate(successive_outputs, axis=0)

"""
用keras实现一个简单RNN
"""
max_features = 10000  # 作为特征的单词数量，也就是高频出现的10000个词语
maxlen = 500  # 评论500词以上截断
batch_size = 32  # 每个批次的数据个数

print('Loading data...')
(input_train, y_train), (input_test, y_test) = load_local(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)  # 填充序列是所有序列都是相同长度
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# model = Sequential()
# model.add(Embedding(max_features, 32))  # 词嵌入
# model.add(SimpleRNN(32))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
# show_acc_results(history)

"""
LSTM实现
"""
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
show_acc_results(history)

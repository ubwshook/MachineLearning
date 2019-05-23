"""
将文本数据向量化的方法之一one-hot的4个示例
1.单词级one-hot
2.字符级one-hot
3.keras API实现one-hot
4.散列实现单词级one-hot
"""
import numpy as np
import string
from keras.preprocessing.text import Tokenizer


def example_one_hot_word(samples, max_length):
    """
    单词级的one-hot，每个单词分配一个索引，将单词表示为长度为max_length且仅在
    单词对应索引上为1，其余位置为0的序列
    :param samples: 输入的字符串列表
    :param max_length: 不同单词的最大个数
    :return: one-hot矩阵
    """
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.

    return results


def example_one_hot_char(samples, max_length):
    """
    字符级one-hot，将所有字符赋予索引值，并进行one-hot编码
    :param samples: 输入的字符串列表
    :param max_length: 不同字符的最大个数
    :return: one-hot矩阵
    """
    characters = string.printable
    # 书中代码疑似有问题，书中方式会使下面token_index.get(character)
    token_index = dict(zip(characters, range(1, len(characters) + 1)))
    results = np.zeros((len(samples), max_length, len(characters) + 1))
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            results[i, j, index] = 1.

    return results


def keras_one_hot(samples):
    """
    Keras API 实现one-hot
    :param samples: 输入的字符串列表
    :return: 语句对应的索引序列，one-hot编码结果
    """
    tokenizer = Tokenizer(num_words=1000)  # 分词器，处理前1000高频出现的单词
    tokenizer.fit_on_texts(samples)  # 创建单词索引
    sequences = tokenizer.texts_to_sequences(samples)  # 把单词转换为序列
    one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # one_hot编码
    word_index = tokenizer.word_index # 单词索引表
    print('Found %s unique tokens.' % len(word_index))
    return sequences, one_hot_results


def hash_one_hot(samples, max_length, dimensionality):
    """
    哈希one-hot，并不为单词固定索引，而是将单词散列编码为长度固定的向量。
    :param samples: 输入的字符串列表
    :param max_length: 不同字符的最大个数
    :param dimensionality: 散列的维度
    :return: hash one-hot结果
    """
    results = np.zeros((len(samples), max_length, dimensionality))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = abs(hash(word)) % dimensionality
            results[i, j, index] = 1.

    return results


max_length = 10
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
dimensionality = 1000
results = example_one_hot_word(samples, max_length)
results = example_one_hot_char(samples, 50)
sequences, one_hot_results = keras_one_hot(samples)
results = hash_one_hot(samples, max_length, dimensionality)




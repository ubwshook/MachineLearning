import numpy as np


def vectorize_sequences(squences, dimension=10000):
    """
    @函数功能:将序列向量化，初始化全0的序列，在单词索引对应的位置上置1
    """
    resluts = np.zeros((len(squences), dimension))
    for i, sequence in enumerate(squences):
        resluts[i, sequence] = 1
    return resluts

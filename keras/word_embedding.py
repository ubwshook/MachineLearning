"""
词嵌入法实现低维度的文本向量化，主要两种方法：
1.完成主任务的同时完成词嵌入
2.使用与计算好的诶嵌入加载到模型中。即预训练词嵌入。

"""
from imdb import load_local
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
import matplotlib.pyplot as plt
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def show_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def simple_embedding():
    """
    一个简单词嵌入的例子，使用的imdb数据集
    :return:
    """
    max_features = 10000
    maxlen = 20  # 在20额单词后截断文本
    (x_train, y_train), (x_test, y_test) = load_local(num_words=max_features)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)  # 整数列表转化为二维数组
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

    model = Sequential()
    model.add(Embedding(10000, 8, input_length=maxlen))  # 参数1是标记的个数，参数2是嵌入的维度，参数3是输入的最大长度
    model.add(Flatten())  # 进入分类器之前要展平
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    show_results(history)


def raw_data_pro():
    """
    原始数据处理，获取文本序列和标签序列
    :return:
    """
    imdb_dir = 'E:/git_code/data/aclImdb'
    train_dir = os.path.join(imdb_dir, 'train')
    labels = []
    texts = []
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), encoding='utf-8')
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    return texts, labels


def tokenizing_data(texts, labels, maxlen, max_words, training_samples, validation_samples):
    """
    对文本序列和label进行分词、向量化得处理，为进入神经网络做好准备。
    :param texts: 文本序列
    :param labels: 标签序列
    :param maxlen: 评论截断的最大长度
    :param max_words: 处理出现频率最高的max_words个词语
    :param training_samples: 在多少个词上进行训练
    :param validation_samples: 在多少个词上进行验证
    :return: 返回训练数据、验证数据、词索引
    """
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)  # 获取分词索引
    sequences = tokenizer.texts_to_sequences(texts)  # 将文本转换为数字序列

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=maxlen)  # 填充或截断序列使其长度董伟maxlen
    labels = np.asarray(labels)  # 标签数据向量化
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)   # 将数据打乱，因为原先正面和负面是排好序的
    data = data[indices]
    labels = labels[indices]
    x_train = data[:training_samples]
    y_train = labels[:training_samples]
    x_val = data[training_samples: training_samples + validation_samples]
    y_val = labels[training_samples: training_samples + validation_samples]

    return x_train, y_train, x_val, y_val, word_index


def get_glove(embedding_dim, word_index):
    """
    加载Glove词嵌入文件
    :param embedding_dim: 嵌入的维度
    :param word_index: 词索引列表
    :return: GloVe词嵌入矩阵
    """
    glove_dir = 'E:/git_code/data/glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # 组装词嵌入矩阵
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

    return embedding_matrix


def pretrained_embedding():
    """
    使用预训练的词嵌入处理IMDB评论情感分类问题
    :return:
    """
    texts, labels = raw_data_pro()
    x_train, y_train, x_val, y_val, word_index = tokenizing_data(texts, labels, maxlen, max_words,
                                                                 training_samples, validation_samples)
    embedding_matrix = get_glove(embedding_dim, word_index)

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.layers[0].set_weights([embedding_matrix])  # 嵌入矩阵
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
    model.save_weights('pre_trained_glove_model.h5')
    show_results(history)


if __name__ == '__main__':
    #simple_embedding()

    maxlen = 100
    training_samples = 200
    validation_samples = 10000
    max_words = 10000
    embedding_dim = 100

    pretrained_embedding()




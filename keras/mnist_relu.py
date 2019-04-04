"""
MNIST数据集是机器学习领域的一个经典数据集，下面这段代码相当于是深度学习的一个hello word。
这个问题可以描述为：将手写数字的灰度图像(28像素 * 28像素)划分到10个类别中(0到9)
"""

from keras.datasets import mnist
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.utils import to_categorical

'''
加载训练数据和测试数据
train_images是图片，train_labels是图片的数字
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)

digit = train_images[4]

'''画出一个图作为示例'''
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

'''
开始为神经网络添加层：
第一层是一个有512个节点，使用relu激活函数，输入为 28 * 28向量的全连接层(dense)
第二层是一个具有是个节点，使用'softmax'激活的全连接层，作为输出层，对应10中输出
'''
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

'''编译网络, 优化器使用rmsprop, 损失函数使用交叉熵损失函数， 网络评估方式是精确度(accuracy)'''
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

'''把训练数据和验证数据都转化为一维张量'''
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

'''将labels转化为向量，在对应位置设置为1'''
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

'''对网络进行训练，进行5轮训练，每批次训练使用128个样本'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)

'''评价网络'''
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)



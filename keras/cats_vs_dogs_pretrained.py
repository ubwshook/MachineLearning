"""
使用VGG16卷积基进行预训练的方法来解决猫狗图片分类的问题
"""

from cats_vs_dogs import create_fold, show_results
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import numpy as np
from keras import layers
from keras import models
from keras import optimizers

""" 
使用本地下载的VGG16卷积基，可以自己下载路径： 
WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
"""
WEIGHTS_PATH_NO_TOP = 'E:\\git_code\\MachineLearning\\keras\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

conv_base = VGG16(weights=WEIGHTS_PATH_NO_TOP, include_top=False, input_shape=(150, 150, 3))


def extract_features(directory, sample_count):
    """
    此函数使用VGG16卷积基对图片进行预处理，提取特征
    :param directory: 数据集路径
    :param sample_count: 使用的样本数
    :return:
    """

    batch_size = 20
    datagen = ImageDataGenerator(rescale=1. / 255)

    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(directory, target_size=(150, 150), batch_size=batch_size,
                                            class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


def bulid_dropout_model():
    """
    构建网络，在聚集基的基础上加上分类器即可，同时加上Dropout
    :return:
    """
    model = models.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model


train_dir, validation_dir, test_dir = create_fold()

""" 特征提取 """
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

""" 构建网络，训练网络 """
model = bulid_dropout_model()
history = model.fit(train_features, train_labels, epochs=30, batch_size=20,
                    validation_data=(validation_features, validation_labels))

show_results(history)



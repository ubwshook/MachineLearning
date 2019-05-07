"""
kaggle猫狗图片分类分问题
数据来源： www.kaggle.com/c/dogs-vs-cats/data
这段代码是一个简单解决方式，它的精度不是很高大概0.7到0.8之间
"""
import os
import shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def create_fold():
    """
    将原始数据中的图片分成训练集、验证集、测试集，并分文件夹存放。
    """
    original_dataset_dir = 'E:\\git_code\\data\\dogs-vs-cats\\train'  # 原始数据的目录

    base_dir = 'E:\\git_code\\data\\cats_and_dogs_small'  # 从原始数据中分裂出来的笑的数据集
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, 'train')  # 创建训练集目录
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    validation_dir = os.path.join(base_dir, 'validation')  # 创建验证集目录
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)

    test_dir = os.path.join(base_dir, 'test')  # 创建测试集目录
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, 'cats')  # 创建cats的训练集目录
    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)

    train_dogs_dir = os.path.join(train_dir, 'dogs')  # 创建dogs的训练集目录
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, 'cats')  # 创建cats的验证集目录
    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)

    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # 创建dogs的验证集目录
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, 'cats')  # 创建cats的测试集目录
    if not os.path.exists(test_cats_dir):
        os.mkdir(test_cats_dir)

    test_dogs_dir = os.path.join(test_dir, 'dogs')  # 创建dogs的测试集目录
    if not os.path.exists(test_dogs_dir):
        os.mkdir(test_dogs_dir)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    return train_dir, validation_dir, test_dir


def bulid_model():
    """
    构建网络：使用Conv2D和MaxPooling2D层交叠构成。
    Flatten层将3D输出展平到1D
    二分类问题最终使用sigmod激活
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    return model


def create_generator(dir):
    """
    创建数据生成器，这个生成器的作用是，将JPEG解码为RGB像素网格，然后将这些像素网格转换为浮点数向量，
    然后将像素值(0~255范围内)缩放到[0,1]区间。
    :param dir: 数据所在的目录
    :return: 返回一个生成器
    """
    dir_datagen = ImageDataGenerator(rescale=1. / 255)
    generator = dir_datagen.flow_from_directory(dir, target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
    return generator


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

if __name__=="__main__":
    """ 创建训练和验证集的目录 """
    train_dir, validation_dir, test_dir = create_fold()
    """ 构建网络 """
    model = bulid_model()

    """ 实例化训练生成器和验证生成器 """
    train_generator = create_generator(train_dir)
    validation_generator = create_generator(validation_dir)

    """ 生成器方式训练网网络 """
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator,
                                  validation_steps=50)
    """ 显示训练结果 """
    show_results(history)











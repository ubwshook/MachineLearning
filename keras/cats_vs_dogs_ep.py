"""
同时使用数据增强和VGG16预训练
提示：这个训练在CPU上回运行的非常慢，最后可以GPU进行运算
"""
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from cats_vs_dogs import create_fold, show_results
import matplotlib.pyplot as plt


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


def build_model():
    """ 构建网络，使用VGG16卷积基 """
    conv_base = VGG16(weights=WEIGHTS_PATH_NO_TOP, include_top=False, input_shape=(150, 150, 3))
    conv_base.traniable = False   # 将卷积基设置为不可训练，不要改变它的权重
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def smooth_curve(points, factor=0.8):
    """
    平滑数据点位，使用指数移动平均值
    :param points: 数据点
    :param factor: 平移因子
    :return: 平滑后的数据点
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


def plot_result_curve(history):
    """
    绘制平滑后的精度和损失图像
    :param history: 训练结果
    :return: 无
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs,
             smooth_curve(acc), 'bo', label='Smoothed training acc')
    plt.plot(epochs,
             smooth_curve(val_acc), 'b', label='Smoothed validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs,
             smooth_curve(loss), 'bo', label='Smoothed training loss')
    plt.plot(epochs,
             smooth_curve(val_loss), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    fine_tunning = True
    train_dir, validation_dir, test_dir = create_fold()
    conv_base = VGG16(weights=WEIGHTS_PATH_NO_TOP, include_top=False, input_shape=(150, 150, 3))

    """ 数据增强 """
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20,
                                                            class_mode='binary')

    """ 构建、编译、训练网络 """
    model = build_model()

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator,
                                  validation_steps=50)

    if fine_tunning:
        conv_base.trainable = True
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])
        history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                                      validation_data=validation_generator,
                                      validation_steps=50)

    """ 显示训练结果 """
    if fine_tunning:
        plot_result_curve(history)
    else:
        show_results(history)

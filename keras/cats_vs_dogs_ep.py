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


train_dir, validation_dir, test_dir = create_fold()

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
""" 显示训练结果 """
show_results(history)

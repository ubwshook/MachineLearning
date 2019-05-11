from cats_vs_dogs import create_fold, show_results
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers


def show_enhanced_image(datagen, image_dir):
    """
    :函数功能: 显示数据增强后的图片
    :param datagen: 数据生成器
    :param image_dir: 图片文件夹
    :return:
    """
    fnames = [os.path.join(image_dir, fname) for
              fname in os.listdir(image_dir)]
    img_path = fnames[2]
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break

    plt.show()

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
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    return model


if __name__ == "__main__":
    """ 创建训练和验证集的目录 """
    train_dir, validation_dir, test_dir = create_fold()
    train_cats_dir = os.path.join(train_dir, 'cats')
    datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
                                 height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode='nearest')

    """ 打印增强的图片 """
    show_enhanced_image(datagen, train_cats_dir)

    """ 使用数据增强的方式，训练网络 """
    model = bulid_model()
    train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=40,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                        batch_size=32, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150),
                                                            batch_size=32, class_mode='binary')
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                                  validation_data=validation_generator, validation_steps=50)

    model.save('cats_and_dogs_small_2.h5')
    show_results(history)


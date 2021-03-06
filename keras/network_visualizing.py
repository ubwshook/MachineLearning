"""
三种可视化卷积神经网络的方法:
1.可视化卷积神经网络的中间输出
2.可视化卷积神经网络的过滤器
3.可视化图像中类激活的热力图
"""
from keras import models
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
import cv2

WEIGHTS_PATH_NO_TOP = 'E:\\git_code\\MachineLearning\\keras\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
WEIGHTS_PATH = 'E:\\git_code\\MachineLearning\\keras\\vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def visualizing_activations(model, img_path):
    """
    可视化激活层
    :return:
    """
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # 增加维度使其成为一个4维张量，符合模型的输入，第一维是图片的索引
    img_tensor /= 255.  # 输入的图片都要进行归一化处理

    print(img_tensor.shape)

    plt.cla()
    plt.close("all")
    plt.imshow(img_tensor[0])
    plt.show()

    layer_outputs = [layer.output for layer in model.layers[:8]]  # 提取模型的每一层输出
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)  # 实例化模型

    activations = activation_model.predict(img_tensor)  # 将图片张量送到模型中进行进行激活
    first_layer_activation = activations[0]
    # 将第4个通道可视化
    plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
    plt.show()

    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()


def deprocess_image(x):
    # 对张量做标准化，使其均值为0，标准差为0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # 将x裁剪到[0,1]区间
    x += 0.5
    x = np.clip(x, 0, 1)

    # 将x转换为RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(model, layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])  # 获取损失
    grads = K.gradients(loss, model.input)[0]  # 获取损失相对于输入的梯度
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])  # 指定输入输出构造一个迭代函数
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.

    # 进行40轮的梯度下降，构造出过滤器最大响应的图
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    img = input_img_data[0]

    return deprocess_image(img)


def visualizing_filter(model, layer_name):
    size = 64
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(model, layer_name, i + (j * 8), size=size)
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
    plt.figure(figsize=(20, 20))
    plt.imshow(results/255.)
    plt.show()


def Visualizing_heatmaps(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    max_index = np.argmax(preds[0])

    predict_output = model.output[:, max_index]
    last_conv_layer = model.get_layer('block5_conv3')  # 选用最后一个卷积层

    # 类别相对于block5_conv3输出特征的梯度
    grads = K.gradients(predict_output, last_conv_layer.output)[0]
    # 其中每个元素是特定特征通道的梯度平均大小
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # 创建迭代器
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    # 将特征图数组的每个通道乘以这个通道对类别的重要程度
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    # 得到的特征图的逐通道平均值，即为类激活的热力图
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()

    # 使用openCV将热力图和原图进行叠加
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite('Rottweiler_heatmap.jpg', superimposed_img)


""" 1.可视化激活层 """
np.seterr(divide='ignore', invalid='ignore')
model = models.load_model('cats_and_dogs_small_2.h5')
model.summary()
img_path = 'E:/git_code/data/cats_and_dogs_small/train/dogs/dog.544.jpg'
visualizing_activations(model, img_path)

""" 2.可视化神经网络过滤器 """
model = VGG16(weights=WEIGHTS_PATH_NO_TOP, include_top=False, input_shape=(150, 150, 3))
layer_name = 'block3_conv1'
plt.imshow(generate_pattern(model, layer_name, 0))
plt.show()
visualizing_filter(model, layer_name)

""" 3.可视化类激活热力图 """
model = VGG16(weights=WEIGHTS_PATH)
img_path = 'E:/git_code/data/cats_and_dogs_small/train/dogs/dog.544.jpg'
Visualizing_heatmaps(model, img_path)





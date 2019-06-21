import keras
from keras import layers
from keras.preprocessing import sequence
from imdb import load_local
from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'D:\\Program Files (x86)\\Graphviz2.38\\bin'


max_features = 500
max_len = 500

(x_train, y_train), (x_test, y_test) = load_local(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
plot_model(model, show_shapes='True', to_file='model.png')

callbacks = [keras.callbacks.TensorBoard(log_dir='my_log_dir',  # 日志写入的位置
                                         histogram_freq=1,   # 每一轮之后记录激活
                                         embeddings_freq=1,   # 每一轮之后记录嵌入数据
                                         embeddings_data=x_train[:100].astype("float32"),)]
history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2,
                    callbacks=callbacks)

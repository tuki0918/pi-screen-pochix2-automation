# coding: utf-8
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import os
train_data_dir = os.path.join(os.getcwd(), 'datasets')
print(train_data_dir)

# 前処理の設定
img_height = 32
img_width = 32
batch_size = 32

# datasets以下のファイルを前処理する
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size)
print(train_generator.num_classes)

K.set_image_data_format('channels_last')

# MNISTのネットワークを使いまわし（良くない）
input_shape = (img_height, img_width,  3)
model = Sequential()
model.add(Conv2D(batch_size, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(batch_size*2, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_generator.num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam')

# 雑に学習
nb_epochs = 100
model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples,
    epochs = nb_epochs)

# モデルの保存
model.save('model-screen-3ch.h5')

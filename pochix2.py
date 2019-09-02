# coding: utf-8
import os
import time
import picamera
import picamera.array
import cv2
import numpy as np

import RPi.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# 学習モデルロード
model = load_model('model-screen-3ch.h5')

############################################
# ラベルの作成
# TODO: モデル作成時にラベルも別ファイルに出力しておく
train_data_dir = os.path.join(os.getcwd(), 'resources', 'datasets')

img_height = 32
img_width = 32
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size)

labels = dict((v, k) for k, v in train_generator.class_indices.items())
############################################

try:
    with picamera.PiCamera() as camera:
        with picamera.array.PiRGBArray(camera) as stream:
            camera.resolution = (320, 240)
            while True:
                # stream.arrayにRGBの順で映像データを格納
                camera.capture(stream, 'bgr', use_video_port=True)
                img = cv2.cvtColor(stream.array, cv2.COLOR_BGR2RGB)
                # リサイズ＋前処理
                re_img = cv2.resize(img, (img_height, img_width))
                re_img = re_img.astype('float32')
                re_img /= 255
                x = re_img.reshape(1, img_height, img_width, 3)
                # 予測
                x_proba = model.predict(x)
                x_classes = x_proba.argmax(axis=-1)
                state = labels[x_classes[0]]

                # streamをリセット
                stream.seek(0)
                stream.truncate()

                # 特定の条件のみ、ポチポチする
                if state == 'A' or state == 'B':
                    GPIO.output(18, True)
                time.sleep(1)
                GPIO.output(18, False)

except KeyboardInterrupt:
    pass

GPIO.cleanup()

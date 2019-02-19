# -*- coding:utf-8 -*-
"""
@author: LiGe
@file: imgSpeed.py
@time: 2019/2/19 2:30 PM
"""
import time
import mxnet as mx
import cv2
import numpy as np
from PIL import Image
from skimage import io
from keras.preprocessing import image

N = 50


def mxReader(imgPath):
    startTime = time.time()
    for i in range(N):
        img = mx.image.imread(imgPath)
    print('mxReader : {}'.format((time.time() - startTime) / N))
    print(img.shape)


def kerasReader(imgPath):
    startTime = time.time()
    for i in range(N):
        img = image.load_img(imgPath)
        img = image.img_to_array(img)

    print('keras reader : {}'.format((time.time() - startTime) / N))


def scipyReader(imgPath):
    startTime = time.time()
    for i in range(N):
        img = io.imread(imgPath)
    print('scipyReader : {}'.format((time.time() - startTime) / N))


def cvReader(imgPath):
    startTime = time.time()
    for i in range(N):
        img = cv2.imread(imgPath)
    print('cvReader : {}'.format((time.time() - startTime) / N))


def pilReader(imgPath):
    startTime = time.time()
    for i in range(N):
        img = Image.open(imgPath)
    print('pilReader : {}'.format((time.time() - startTime) / N))


def pilReaderArray(imgPath):
    startTime = time.time()
    for i in range(N):
        img = Image.open(imgPath)
        img = np.array(img)
    print('pilReaderArray : {}'.format((time.time() - startTime) / N))


if __name__ == "__main__":
    img_path = ''
    mxReader(img_path)
    cvReader(img_path)
    kerasReader(img_path)
    scipyReader(img_path)
    pilReader(img_path)
    pilReaderArray(img_path)

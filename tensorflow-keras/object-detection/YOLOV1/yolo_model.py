import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D
from tensorflow.keras.regularizers import l2
from glob import glob
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as elementTree
import json
import os

from config import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(f"Tensorflow Version: {tf.__version__}")

class yolov1():
    def __init__(self):
        super(yolov1, self).__init__()

    def CNNBlock(self, x, nfilters, size, stride, pool=None):
        x = Conv2D(filters=nfilters, kernel_size=size, padding='same', strides=stride, kernel_initializer='he_normal',
                   kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        if pool is not None:
            x = MaxPooling2D(pool_size=pool)(x)
        return x



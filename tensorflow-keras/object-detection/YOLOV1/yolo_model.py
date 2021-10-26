import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Input, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
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
    def __init__(self, config):
        super(yolov1, self).__init__()
        self.config = config
        self.architecture = config['model_arch']

    def CNNBlock(self, x, kernel_size, nfilters, stride, padding):
        x = Conv2D(filters=nfilters, kernel_size=kernel_size, padding=padding, strides=stride,
                   kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

    def create_darknet(self, input_shape):
        input_layer = Input(shape=(*input_shape, 3))
        x = input_layer
        for block in self.architecture:
            if block[0] == 'CNN':
                x = self.CNNBlock(x, *block[1:])
            elif block[0] == 'Max':
                x = MaxPooling2D(pool_size=block[1], strides=block[2], padding=block[3])(x)
        x = Flatten()(x)
        return x

    def add_fc_layers(self, x, output_shape):
        x = Dense(512, linear=False)(x)
        x = Dense(5096, linear=False)(x)
        x = Dropout(0.5)(x)
        x = Dense(1470, linear=True)(x)
        return x


class yolo_reshape(tf.keras.layers):
    def __init__(self, config):
        super(yolo_reshape, self).__init__()
        self.target_shape = config['output_shape']
        self.config = config

    def __call__(self, input):
        # get_grid size
        S = [self.target_shape[0], self.target_shape[1]]
        # classes
        C = self.config['n_classes']
        # boxes
        B = self.config['n_boxes']

        idx1 = S[0] * S[1] * C #each block first C values
        idx2 = idx1 + S[0]*S[1]*B

        # class probabilities
        class_prob = K.reshape(input[:, :idx1], (K.shape(input)[0], ) + tuple(S[0], S[1], C))
        class_prob = K.softmax(class_prob)

        # confidence
        confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0], ) + tuple(S[0], S[1], B))
        confs = K.sigmoid(confs)

        # Boxes
        boxes = K.reshape(input[:, idx2:], (K.shape(input)[0], ) + tuple(S[0], S[1], B*4))

        outputs = K.concatenate([class_prob, confs, boxes])

        return outputs


    








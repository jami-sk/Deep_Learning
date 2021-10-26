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

class YoloV1():
    def __init__(self, config):
        super(YoloV1, self).__init__()
        self.config = config
        self.architecture = config['model_arch']

    def cnn_block(self, x, kernel_size, nfilters, stride, padding):
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
                x = self.cnn_block(x, *block[1:])
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


class YoloReshape(tf.keras.layers):
    def __init__(self, config):
        super(YoloReshape, self).__init__()
        self.target_shape = config['output_shape']
        self.config = config

    def __call__(self, input):
        # get_grid size
        S = [self.target_shape[0], self.target_shape[1]]
        # classes
        C = self.config['n_classes']
        # boxes
        B = self.config['n_boxes']

        idx1 = S[0] * S[1] * C  # each block first C values
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


class YoloLoss:
    def __init__(self, y_true, y_pred):
        super(YoloLoss, self).__init__()
        self.y_pred = y_pred
        self.y_true = y_true

    def xywh2minmax(self, xy, wh):
        xy_min = xy - wh / 2
        xy_max = xy + wh / 2
        return xy_min, xy_max

    def iou(self, pred_mins, pred_maxes, true_mins, true_maxes):
        intersect_mins = K.maximum(pred_mins, true_mins)
        intersect_maxes = K.minimum(pred_maxes, true_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        pred_wh = pred_maxes - pred_mins
        true_wh = true_maxes - true_mins
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        return iou_scores

    def yolo_head(self, feats):
        # Dynamic implementation of conv dims for fully convolutional model.
        conv_dims = K.shape(feats)[1:3]  # assuming channels last
        # In YOLO the height index is the inner most iteration.
        conv_height_index = K.arange(0, stop=conv_dims[0])
        conv_width_index = K.arange(0, stop=conv_dims[1])
        conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

        # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
        # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
        conv_width_index = K.tile(
            K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
        conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = K.cast(conv_index, K.dtype(feats))

        conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

        box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
        box_wh = feats[..., 2:4] * 448

        return box_xy, box_wh

    def __call__(self):
        label_class = self.y_true[..., :20]  # ? * 7 * 7 * 20
        label_box = self.y_true[..., 22:24]  # ? * 7 * 7 * 4
        response_mask = self.y_true[..., 24]  # ? * 7 * 7
        response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

        predict_class = self.y_pred[..., :20]  # ? * 7 * 7 * 20
        predict_trust = self.y_pred[..., 20:22]  # ? * 7 * 7 * 2
        predict_box = self.y_pred[..., 22:]  # ? * 7 * 7 * 8

        _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
        _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

        label_xy, label_wh = self.yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
        label_xy_min, label_xy_max = self.xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

        predict_xy, predict_wh = self.yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
        predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
        predict_xy_min, predict_xy_max = self.xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

        iou_scores = self.iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
        best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
        best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

        box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

        no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
        object_loss = box_mask * response_mask * K.square(1 - predict_trust)
        confidence_loss = no_object_loss + object_loss
        confidence_loss = K.sum(confidence_loss)

        class_loss = response_mask * K.square(label_class - predict_class)
        class_loss = K.sum(class_loss)

        _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
        _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

        label_xy, label_wh = self.yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
        predict_xy, predict_wh = self.yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

        box_mask = K.expand_dims(box_mask)
        response_mask = K.expand_dims(response_mask)

        box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
        box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
        box_loss = K.sum(box_loss)

        loss = confidence_loss + class_loss + box_loss

        return loss











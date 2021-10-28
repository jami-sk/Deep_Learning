__all__ = ['LoadData']
import xml.etree.ElementTree as elementTree
import json
from glob import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import pandas as pd


from config import config


class LoadData:
    def __init__(self, images_path, labels_path, first_time=False):
        """
            Load Images and XML files and test image and xml correspondence and test image and label correspondence
            Input Params:
                :param images_path: images folder
                :param labels_path: xml files folder
        """
        super(LoadData, self).__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_files = glob(f"{self.images_path}/{'*.jpg'}")
        self.lbl_files = glob(f"{self.labels_path}/{'*.xml'}")
        self.W = config['new_size'][1]
        self.H = config['new_size'][0]
        self.N = len(self.img_files)
        self.C = config['n_classes']
        self.S = config['grid_size']
        self.B = config['n_boxes']

        # create data csv file
        if first_time:
            self.create_image_csv()

    def create_image_csv(self):
        df_list = list()
        for idx, img_file in enumerate(self.img_files):
            image = cv2.imread(img_file)
            img_shape = image.shape
            df_list.append([img_file.split('\\')[-1], img_file, self.lbl_files[idx], img_shape])
        df = pd.DataFrame(data=df_list, columns=['Image', 'File_Path', 'XML_Path', 'Original_Size'])
        df.to_csv('./data/images.csv')

    @staticmethod
    def read_xml(path) -> [dict]:
        """
            Read XML files for object detection and returns image size and list objects in the image
            Reads one image at once wrt the index provided
            Input Params:
                :param path: file path of the label xml
            Output Params:
                :return: returns image size and list of objects and bounding boxes
        """
        tree = elementTree.parse(path)
        root = tree.getroot()
        object_list = list()
        for obj in root.iter('object'):
            object_dict = dict()
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in config['classes'] or int(difficult) == 1:
                continue
            cls_id = config['classes'].index(cls)
            xml_box = obj.find('bndbox')
            b = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text),
                 int(xml_box.find('xmax').text), int(xml_box.find('ymax').text))
            object_dict['cls_id'] = cls_id
            object_dict['class'] = cls
            object_dict['BB'] = b
            object_list.append(object_dict)
        return object_list

    @staticmethod
    def read_image(path):
        # image = load_img(path, color_mode='rgb')
        # image = img_to_array(image)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.shape[:2], image

    def preprocess_image(self, image, label):
        img, lbl = cv2.resize(image, (self.W, self.H)), label
        return img, lbl

    def convert_boxes_label(self, path, height, width):
        object_list = self.read_xml(path)
        out_label = np.zeros(config['output_shape'], dtype=np.float32)
        for obj_dict in object_list:
            cls_idx = obj_dict['cls_id']
            x1, y1, x2, y2 = obj_dict['BB']
            x1, y1, x2, y2 = x1 * self.W / width, y1 * self.H / height, x2 * self.W / width, y2 * self.H / height
            x = ((x1 + x2) / 2) / (self.W / self.S[1])
            y = ((y1 + y2) / 2) / (self.H / self.S[0])
            w = (x2 - x1) / self.W
            h = (y2 - y1) / self.H
            int_x = int(x)
            int_y = int(y)
            delta_x, delta_y = x - int_x, y - int_y
            out_label[int_y, int_x, cls_idx] = 1
            out_label[int_y, int_x, self.C] = 1
            out_label[int_y, int_x, self.C + self.B:self.C + self.B + 4] = [delta_x, delta_y, w, h]

        return out_label

    def convert_label_boxes(self, label):
        boxes = label[..., self.C + self.B:]
        boxes_shape = boxes.shape[:2]
        boxes = boxes.reshape(*label.shape[:2], self.B, 4)
        out_boxes = boxes.copy()
        height_index = np.arange(0, stop=self.S[0], dtype=float)
        height_index = np.tile(np.expand_dims(height_index, 1), [self.S[1]])
        width_index = np.arange(0, stop=self.S[1], dtype=float)
        width_index = np.tile(np.expand_dims(width_index, 0), [self.S[0], 1])
        wh_index = np.transpose(np.stack([height_index, width_index]))
        wh_index = wh_index.reshape([*self.S, 1, 2])
        label_shape = np.array(boxes_shape).reshape([1, 1, 1, 2])
        out_boxes[..., :2] = (out_boxes[..., :2] + wh_index) / label_shape * config['new_size'][0]
        out_boxes[..., 2:4] = out_boxes[..., 2:4] * self.H
        out_boxes[..., :2] -= out_boxes[..., 2:4] / 2
        object_list = list()
        for row_idx in range(self.S[0]):
            for col_idx in range(self.S[1]):
                for box_idx in range(self.B):
                    if label[row_idx, col_idx, self.C + box_idx] > 0.5:
                        cls_idx = label[row_idx, col_idx, :self.C].max()
                        bb = out_boxes[row_idx, col_idx, box_idx, :]
                        object_list.append({'cls_idx': cls_idx, 'BB': bb})
        return object_list

    def test_image_xml(self):
        rand_int = np.random.randint(len(self.img_files))
        _, image = self.read_image(self.img_files[rand_int])
        object_list = self.read_xml(self.lbl_files[rand_int])
        f, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image)
        for obj in object_list:
            bb = obj['BB']
            rect = Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(bb[0], bb[1], s=f"{obj['class']}", color='white')
        plt.show()

    def test_image_label(self):
        rand_int = np.random.randint(len(self.img_files))
        image_shape, image = self.read_image(self.img_files[rand_int])
        label_matrix = self.convert_boxes_label(self.lbl_files[rand_int], *image_shape)
        image, lbl = self.preprocess_image(image, label_matrix)
        out_object_list = self.convert_label_boxes(label_matrix)
        f, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image)
        for obj in out_object_list:
            bb = obj['BB']
            rect = Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(bb[0], bb[1], s=f"{obj['cls_idx']}", color='white')
        plt.show()



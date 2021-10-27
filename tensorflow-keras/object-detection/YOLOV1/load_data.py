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


from config import config


class LoadData:
    def __init__(self, images_path, labels_path):
        super(LoadData, self).__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self.img_files = glob(f"{self.img_path}/{'*.jpg'}")
        self.lbl_files = glob(f"{self.label_path}/{'*.xml'}")
        self.N = len(self.img_files)
        self.C = config['n_classes']
        self.S = config['grid_size']
        self.B = config['n_boxes']

    @staticmethod
    def read_xml(path) -> list(dict):
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
        image = load_img(path, color_mode='rgb')
        image = img_to_array(image)
        return image

    @staticmethod
    def preprocess_image(image,label):
        img, lbl = image, label
        return img,lbl

    def convert_boxes_label(self, path):
        object_list = self.read_xml(path)
        out_label = np.zeros(config['output_shape'],dtype=np.float32)
        for obj_dict in object_list:
            cls_idx = obj_dict['cls_id']
            bb = obj_dict['BB']
            x = ((bb[0]+bb[2])/2)/(config['new_size'][1]/config['grid_size'][1])
            y = ((bb[1]+bb[3])/2)/(config['new_size'][0]/config['grid_size'][0])
            w = (bb[2]-bb[0])/config['new_size'][1]
            h = (bb[3]-bb[1])/config['new_size'][0]
            int_x = int(x)
            int_y = int(y)
            delta_x, delta_y = x-int_x, y-int_y
            out_label[int_y, int_x, cls_idx] = 1
            out_label[int_y, int_x, self.C] = 1
            out_label[int_y, int_x, self.C+self.B:self.C+self.B+4] = [delta_x, delta_y, w, h]

        return out_label

    def convert_label_boxes(self, label):
        object_list = list()
        return object_list
    
    def test_image_xml(self):
        rand_int = np.random.randint(len(self.img_files))
        img_path = self.img_files[rand_int]
        lbl_path = self.lbl_files[rand_int]
        image = self.read_image(img_path)
        object_list = self.read_xml(lbl_path)
        f,ax = plt.subplots()
        ax.imshow(image)
        for obj in object_list:
            bb = obj['BB']
            rect = Rectangle(bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text((bb[0],bb[1]), f"{obj['class']}", color='white')
        plt.show()
    
    def test_image_label(self):
        rand_int = np.random.randint(len(self.img_files))
        img_path = self.img_files[rand_int]
        lbl_path = self.lbl_files[rand_int]
        image = self.read_image(img_path)
        label_matrix = self.convert_boxes_label(lbl_path)
        out_object_list = self.convert_label_boxes(label_matrix)
        f,ax = plt.subplots()
        ax.imshow(image)
        for obj in out_object_list:
            bb = obj['BB']
            rect = Rectangle(bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text((bb[0],bb[1]), f"{obj['class']}", color='white')
        plt.show()



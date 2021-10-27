__all__ = []
import xml.etree.ElementTree as elementTree
import json
from glob import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm


from config import config


class LoadData:
    def __init__(self, img_path, label_path):
        super(LoadData, self).__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.img_files = glob(f"{self.img_path}/{'*.jpg'}")
        self.lbl_files = glob(f"{self.label_path}/{'*.xml'}")
        self.N = len(self.img_files)

    def read_xml(self):
        return

    def read_image(self):
        return

    def convert_boxes_label(self):
        return

    def test_label(self):
        return

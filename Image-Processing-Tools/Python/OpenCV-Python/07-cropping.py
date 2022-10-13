# Usage
# python 07-cropping.py -i <path of the image.
# python 07-cropping.py -i D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png

import argparse
import cv2 
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Pah of the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)


crop = image[150:,:]
cv2.imshow("OpenCV Text", crop)

logo = image[:150,:]
cv2.imshow("OpenCV Circles", logo)
cv2.waitKey(0)

# Usage
# python translate.py -i <D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png>

import numpy as np
import argparse
import imutils
import cv2

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--image", help="Path of the image file")
args = vars(arg_parser.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)

# shift the image 25 pixels to the right and 50 pixels down
# x direction 25 pixesl
# y direction 50 pixels

# Translation Matrix Format
# [
#   [1,0,shiftX],
#   [0,1,shiftY]
# ]

# x --> Positive ---> shift right
# x --> Negative ---> shift left
# y --> Positive ---> shift down
# y --> Negative ---> shift up
M = np.float32([[1,0,25], [0,1,50]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("Translated by 25 Right and 50 Down", shifted)

shifted = imutils.translate(image, 25, 50)
cv2.imshow("Translation using imutils", shifted)
cv2.waitKey(0)

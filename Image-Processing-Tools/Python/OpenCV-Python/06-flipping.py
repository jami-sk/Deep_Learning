# Usage
# python 06-flipping.py -i <path to image>
# python 06-flipping.py -i D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png

import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Path of the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)

# Flip image Horizontally
hor_flip = cv2.flip(image, 1)
cv2.imshow("H Flip", hor_flip)

# Flip Image Vertically
vert_flip = cv2.flip(image, 0)
cv2.imshow("V Flip", vert_flip)

# Flip in both directions
flip = cv2.flip(image, -1)
cv2.imshow("Flip Both Sides", flip)
cv2.waitKey(0)
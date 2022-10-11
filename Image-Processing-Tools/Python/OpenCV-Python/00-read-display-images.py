# Script Usage
# read-display-images.py -i <image file path>
# read-display-images.py -i D:\Development\Deep_Learning\Python-Basics\Sample-Images\Range-Rover-Velar.png

import argparse
import cv2

#constructing argument parser and parse the arguments
arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-i", "--image", required=True, help="Path of the image file")
args = vars(arg_parse.parse_args())

#load the image from the disk using cv2.imread()
image = cv2.imread(args["image"])
(h,w,c) = image.shape[:3]

print(f"image loaded of size (height x width x channels) : {h} x {w} x {c}")

# show the image
cv2.imshow("Loaded Image", image)
cv2.waitKey(0)

# save the image to the disk

cv2.imwrite(".\new_image.jpg", image)

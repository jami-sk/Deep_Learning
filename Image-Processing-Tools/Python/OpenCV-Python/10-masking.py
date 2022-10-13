# Usage
# python 10-masking.py -i <path of the image>
# python 10-masking.py -i D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png

import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Pah of the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)

# Create Mask
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0,150), (image.shape[1]-1,image.shape[0]-1), 255, -1)
cv2.imshow("Reactangular Mask", mask)


# Masking using bitwise operations
masked  = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked Image", masked)
cv2.waitKey(0)

# Create Circular Mask
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(mask, (image.shape[1]//2-8,180//2), 76, 255, -1)
cv2.imshow("Reactangular Mask", mask)

# Masking using bitwise operations
masked  = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Masked Image", masked)
cv2.waitKey(0)



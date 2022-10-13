# Usage
# python 11-split-merge-channels.py -i <path of the image>
# python 11-split-merge-channels.py -i D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png

import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="Path of the Image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\Opencv-Logo.png")
args = vars(ap.parse_args())

image=cv2.imread(args["image"])
cv2.imshow("Loaded Image", image)

# Splitting into channels
(B,G,R) = cv2.split(image)

# Show each channel
cv2.imshow("Red",R)
cv2.imshow("Green", G)
cv2.imshow("Blue", B)

# Merging the channels
merged = cv2.merge([B,G,R])
cv2.imshow("Merged", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# visualize each channel in color
zeros = np.zeros(image.shape[:2], dtype="uint8")
cv2.imshow("Red",cv2.merge([zeros, zeros, R]))
cv2.imshow("Green",cv2.merge([zeros, G, zeros]))
cv2.imshow("Blue",cv2.merge([B, zeros, zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()


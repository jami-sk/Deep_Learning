# Usage
# python 19-edge-detection.py -i <path to the image>

import cv2 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to the image", default = "D:\Development\Deep_Learning\Image-Processing-Tools\Python\Sample-Images\coins.jpg")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)

cv2.imshow("Original", image)
cv2.imshow("Blurred", blurred)

# Compute a "wide", "mid-range", and "tight" thresholds for the edges
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30,150)
tight = cv2.Canny(blurred, 240,250)

# show outputs of canny edge 
cv2.imshow("Wide Egde Map", wide)
cv2.imshow("Mid Edge Map", mid)
cv2.imshow("Tight Edge Map", tight)
cv2.waitKey(0)
cv2.destroyAllWindows()
